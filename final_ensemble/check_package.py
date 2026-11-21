"""Check the exported inference ZIP's structure and its actual offline CPU interface."""
import argparse
import ast
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import sys
import tempfile
from zipfile import ZipFile


STATIC_SHA256 = {
    "models/blip/config.json": "3e6464c2ce7c54512ddb101c5e9a8e77f4c2d637be9e3d005667ccd4a34c6ef2",
    "models/blip/preprocessor_config.json": "0aa66e2e9ac3ea3b5cd4388c35072e22db4e1cc1f96c7872bed07749c712ade1",
    "models/blip/special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "models/blip/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "models/blip/tokenizer_config.json": "86da6fdb761b02f73a05561aba71711c2d7c205fe1fd1744046173a410263925",
    "models/blip/vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "models/siglip/config.json": "cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688",
    "models/siglip/preprocessor_config.json": "d11ccb80f15d358a11bdb070e92e2d889005874b7db15823d5f10d9b2533b14a",
    "models/siglip/special_tokens_map.json": "2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26",
    "models/siglip/spiece.model": "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec",
    "models/siglip/tokenizer.json": "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920",
    "models/siglip/tokenizer_config.json": "d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"
}
SCHEMAS = {
    "models/blip/model.safetensors": "27924de45e0843099b1106b8fca49e6972f58b6e197477daf2f0938ee6254ed4",
    "models/siglip/model.safetensors": "87895c11304954490939db92eb8e7df5ce913aa03d441c5ae6baf769340b31ff",
    "models/pair_head.safetensors": "9d2f3e3debcf19715dc3960a71e8ebd3f750c44e0c754289430ddbd7febe6659",
}
REQUIRED = set(STATIC_SHA256) | {"model.py", "requirements.txt", "ensemble_config.json",
        "models/blip/model.safetensors", "models/siglip/model.safetensors", "models/arms/S0.safetensors"}
OPTIONAL = {"models/arms/C.safetensors", "models/arms/S1_noaux.safetensors", "models/pair_head.safetensors"}
ENTRIES = {"load_model", "predict", "prepare_gallery", "retrieve_candidates", "score_candidates"}


def require(condition, why):
    if not condition:
        raise ValueError(why)


def unique_object(pairs):
    out = dict(pairs)
    require(len(out) == len(pairs), "Duplicate JSON field")
    return out


def sha(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def code_check(source):
    parsed = ast.parse(source)
    found = {n.name for n in parsed.body if isinstance(n, ast.FunctionDef)}
    require(ENTRIES <= found, f"Missing export functions: {sorted(ENTRIES - found)}")


def check_pins(data):
    seen = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9][A-Za-z0-9_.+!-]*)", line)
        require(match, "Expected a single pinned PyPI requirement per nonempty line")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        require(name not in seen and name not in {"torch", "torchvision", "packaging"}, "Duplicate or immutable dependency")
        seen[name] = version
    require(seen.get("transformers") == "4.51.3", "Unexpected Transformers pin for this inference implementation")
    return seen


def check_config(data, names):
    cfg = json.loads(data, object_pairs_hook=unique_object)
    require(set(cfg) == {"arms", "weights", "width", "use_pair"}, "Unexpected ensemble config fields")
    arms = cfg["arms"]
    require(arms in (["S0"], ["S0", "S1_noaux"]), "Unexpected default arm selection")
    require(all(f"models/arms/{arm}.safetensors" in names for arm in arms), "Missing configured arm")
    require(cfg["weights"] == {arm: 1.0/len(arms) for arm in arms} and cfg["width"] == 8
            and cfg["use_pair"] is False, "Unexpected default ensemble recipe")
    return cfg


def tensors(archive, name):
    with archive.open(name) as source:
        raw = source.read(8)
        require(len(raw) == 8, "Missing safetensors header")
        size, = struct.unpack("<Q", raw)
        require(0 < size <= 16 * 1024**2, "Unreasonable safetensors header")
        header = json.loads(source.read(size), object_pairs_hook=unique_object)
    meta = header.pop("__metadata__", {})
    allowed = [{}, {"format": "pt"}] if name.startswith(("models/blip/", "models/siglip/")) else [{}]
    require(meta in allowed, "Unexpected safetensors metadata in " + name)
    blocks, schema = [], {}
    for key, tensor in header.items():
        require(set(tensor) == {"dtype", "shape", "data_offsets"}, "Unexpected tensor descriptor")
        kind, shape = tensor["dtype"], tensor["shape"]
        sizes = {"F32":4, "I64":8}
        require(kind in sizes and isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "Invalid fitted tensor dtype or shape")
        a,b = tensor["data_offsets"]
        require(type(a) is int and type(b) is int and a >= 0 and b-a == sizes[kind]*math.prod(shape), "Invalid tensor extent")
        schema[key] = [kind,shape]
        blocks.append((a,b))
    cursor = 0
    for a,b in sorted(blocks):
        require(a == cursor, "Unexpected gap in tensor storage")
        cursor = b
    require(cursor+8+size == archive.getinfo(name).file_size, "Trailing or missing tensor bytes")
    if name in SCHEMAS:
        fingerprint = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",",":")).encode()).hexdigest()
        require(fingerprint == SCHEMAS[name], "Tensor names or shapes differ from the public architecture in " + name)
    return schema


def inspect_zip(path, out_dir=None):
    require(path.stat().st_size <= 5_000_000_000, "Archive exceeds the official 5 GB limit")
    with path.open("rb") as raw:
        archive_sha = sha(raw)
        raw.seek(0)
        with ZipFile(raw) as z:
            names = set(z.namelist())
            require(len(names) == len(z.namelist()) and REQUIRED <= names and names <= REQUIRED | OPTIONAL,
                    f"Unexpected ZIP contents. Missing {sorted(REQUIRED-names)}, extra {sorted(names-REQUIRED-OPTIONAL)}")
            require(not z.comment, "ZIP comment present")
            require(sum(x.file_size for x in z.infolist()) <= 5_000_000_000, "Unexpected total payload size")
            for entry in z.infolist():
                require(not entry.comment and not entry.flag_bits & (1|8), "Unexpected per-file comment/encryption/descriptor")
                require((entry.external_attr >> 16) & 0o170000 in (0,0o100000), "Non-regular file")
                # All current model members are smaller than 2 GiB. Larger offsets may use structural ZIP64 only.
                offset_payload = struct.pack("<HHQ", 1, 8, entry.header_offset)
                require(entry.extra in (b"", offset_payload), "Unexpected central ZIP extra field")
                raw.seek(entry.header_offset)
                fields = struct.unpack("<4s5H3I2H", raw.read(30))
                require(fields[0] == b"PK\x03\x04" and raw.read(fields[-2]) == entry.filename.encode("ascii"), "Local ZIP entry mismatch")
                local = raw.read(fields[-1])
                zip64_sizes = struct.pack("<HHQQ", 1, 16, entry.file_size, entry.compress_size)
                require(local == b"" or (local == zip64_sizes and fields[7:9] == (0xffffffff,0xffffffff)),
                        "Unexpected local ZIP extra field")
            code_check(z.read("model.py"))
            pins = check_pins(z.read("requirements.txt"))
            config = check_config(z.read("ensemble_config.json"), names)
            for name, wanted in STATIC_SHA256.items():
                with z.open(name) as fp:
                    require(sha(fp) == wanted, "Unexpected public tokenizer or configuration asset: " + name)
            base = tensors(z, "models/blip/model.safetensors")
            for name in sorted(n for n in names if n.endswith(".safetensors") and n != "models/blip/model.safetensors"):
                actual = tensors(z,name)
                if name.startswith("models/arms/"):
                    expected = {k:v for k,v in base.items() if k.endswith(("weight","bias")) and k.startswith(("text_encoder.","itm_head."))}
                    if name.endswith("/C.safetensors"):
                        expected = {k:v for k,v in expected.items() if k.startswith("itm_head.") or ".layer.11.crossattention." in k}
                    require(actual == expected, "Unexpected overlay schema in " + name)
            if out_dir is not None:
                for name in sorted(names):
                    dst = out_dir / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as src, dst.open("wb") as dstfile:
                        shutil.copyfileobj(src, dstfile, 8 << 20)  # at EOF zipfile verifies each member's CRC
            else:
                require(z.testzip() is None, "Archive CRC check failed")
    return {"archive_sha256":archive_sha,"archive_bytes":path.stat().st_size,"entries":len(names),
            "requirements":pins,"config":config,"validated_stage":"structure","crc":True}


def examples(work, image_dir, prompts_path):
    if image_dir is not None:
        images = sorted(Path(image_dir).glob("*.png"))[:8]
        require(len(images) == 8, "--image-dir must contain at least eight PNGs")
        prompts_path = prompts_path or (Path(image_dir).parent / "prompts.jsonl")
        by_id = {}
        if prompts_path.is_file():
            with prompts_path.open(encoding="utf-8-sig") as source:
                for line in source:
                    row = json.loads(line)
                    by_id[str(row.get("id", row.get("prompt_id")))] = row.get("prompts", row.get("prompt"))
        captions = [by_id.get(f.stem) for f in images[:2]]
    else:
        from PIL import Image, ImageDraw
        images = []
        for i in range(8):
            img = Image.new("RGB", (144,112), (60+i*17,180-i*15,73+i*7))
            ImageDraw.Draw(img).rectangle((i*7,18,57+i*10,75), fill=(220-i*25,40+i*12,190))
            path = work / f"{i:04d}.png"
            img.save(path)
            images.append(path)
        captions = [None,None]
    fallbacks = ["A red rectangle near the left side.","A blue shape to the right of a green object."]
    qs = [{"prompt_id":f"q_{i:03d}", "prompt":c or fallbacks[i]} for i,c in enumerate(captions)]
    cs = [{"image_id":p.stem,"image_path":str(p.resolve())} for p in images]
    return qs,cs


def native_margins(model, qs, cs, rows, selected_arms=None):
    """Compare the optimized raw score with each fitted arm's original HF paired forward."""
    from contextlib import ExitStack
    import torch
    from PIL import Image

    paths = {r["image_id"]: r["image_path"] for r in cs}
    samples = [(q["prompt"], rows[q["prompt_id"]][0]) for q in qs]
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(paths[r["image_id"]])).convert("RGB") for _, r in samples]
        for image in images:
            stack.callback(image.close)
        inputs = model["blip_processor"](images=images, text=[t for t,_ in samples],
                                         padding=True, truncation=True, max_length=128, return_tensors="pt")
    blip = model["blip"]
    original = blip.text_encoder, blip.itm_head
    errors = {}
    try:
        with torch.inference_mode():
            for arm, trained_heads in model["heads"].items():
                if selected_arms is not None and arm not in selected_arms:
                    continue
                blip.text_encoder, blip.itm_head = trained_heads
                logits = blip(**inputs, use_itm_head=True, return_dict=True).itm_score
                expected = logits[:,1]-logits[:,0]
                actual = torch.tensor([r["margins"][arm] for _, r in samples])
                require(torch.allclose(expected,actual,atol=0.0003,rtol=0.00005),
                        f"{arm} optimized margins differ from native HF paired forward")
                errors[arm] = (expected-actual).abs().max().item()
    finally:
        blip.text_encoder, blip.itm_head = original
    return errors


def smoke(root, image_dir, prompts_path, work):
    denied = []
    def guard(event, args):
        if event in {"socket.connect","socket.sendto","socket.sendmsg","socket.getaddrinfo","socket.gethostbyname","socket.gethostbyaddr"}:
            denied.append(event)
            raise RuntimeError("External network disabled during QA")
    sys.addaudithook(guard)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(work/"hf-empty"),
                      USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    qs,cs = examples(work,image_dir,prompts_path)
    spec = util.spec_from_file_location("validated_final_model", root/"model.py")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0,str(root))
    spec.loader.exec_module(module)
    model = module.load_model(str(root), "cpu")
    default_weights = model["weights"]
    batched = module.predict(model,qs,cs,top_k=5)
    gallery = module.prepare_gallery(model,cs)
    pool = module.retrieve_candidates(model,qs,gallery,top_k=8)
    rows = module.score_candidates(model,qs,gallery,pool)
    require(set(batched) == {q["prompt_id"] for q in qs} == set(rows) == set(pool), "Query IDs changed")
    known = {c["image_id"] for c in cs}
    # Reuse the already encoded gallery for per-query checks. predict above used the starterkit argument type.
    for q in qs:
        key = q["prompt_id"]
        require(len(batched[key]) == len(set(batched[key])) == 5 and set(batched[key]) <= known, "Invalid Top-5")
        require(len(pool[key]) == len(set(pool[key])) == 8 and set(pool[key]) <= known, "Invalid retrieval shortlist")
        row = rows[key]
        require([x["rank"] for x in row] == list(range(1,9)) and {x["image_id"] for x in row} == set(pool[key]), "Invalid scoring rows")
        require(all(math.isfinite(v) for x in row for v in [x["score"],x["siglip_cosine"],*x["margins"].values()]), "Nonfinite ensemble score")
        require(all(set(x["margins"]) == set(model["heads"]) for x in row), "Missing arm margin")
        require([x["image_id"] for x in row[:5]] == batched[key], "Common-pool scores differ from predict")
        alone = module.predict(model,[q],gallery,top_k=5)
        require(alone[key] == batched[key], "Single and batched rankings differ")
    native = native_margins(model,qs,cs,rows)
    pair_checked = False
    ac_checked = False
    if (root/"models"/"pair_head.safetensors").is_file():
        import gc
        del model,gallery
        gc.collect()
        has_c = (root/"models"/"arms"/"C.safetensors").is_file()
        arms = [*default_weights,"A","C"] if has_c else None
        paired = module.load_model(str(root), "cpu", arms=arms, use_pair=True)
        pair_gallery = module.prepare_gallery(paired,cs)
        with_pair = module.score_candidates(paired,qs,pair_gallery,pool,weights=default_weights,use_pair=True)
        for q in qs:
            key = q["prompt_id"]
            before = [r["image_id"] for r in rows[key]]
            after = [r["image_id"] for r in with_pair[key]]
            require(after[2:] == before[2:] and sorted(after[:2]) == sorted(before[:2]),
                    "Pair comparator changed the pool outside the first two")
            top = sorted(with_pair[key][:2], key=lambda r:r["rank_before_pair"])
            probs = [r["pair_win_probability"] for r in top]
            require([r["rank_before_pair"] for r in top] == [1,2]
                    and all(0 <= p <= 1 for p in probs) and abs(sum(probs)-1) < 1e-5,
                    "Invalid pair evidence")
            require(after[0] == before[int(probs[1] >= 0.75)], "Pair threshold mismatch")
        pair_checked = True
        if has_c:
            import numpy as np
            key = qs[0]["prompt_id"]
            subpool = {key: pool[key][:3]}
            ab = module.score_candidates(paired,qs[:1],pair_gallery,subpool,
                                         weights={"A":0.5,"C":0.5},use_pair=False)
            native.update(native_margins(paired,qs[:1],cs,ab,{"A","C"}))
            values = np.array([[r["margins"][arm] for r in ab[key]] for arm in ("A","C")])
            z = (values-values.mean(axis=1,keepdims=True))/np.maximum(values.std(axis=1,keepdims=True),1e-8)
            require(np.allclose([r["score"] for r in ab[key]],z.mean(axis=0),atol=1e-7),
                    "A+C mixture differs from mean standardized margins")
            ac_checked = True
    require(not denied, "An external network connection was attempted")
    return {"validated_stage":"fresh_offline_cpu_inference","query_count":len(qs),"candidate_count":len(cs),
            "sample_type":"provided_pngs" if image_dir else "synthetic_pngs",
            "starterkit_interface":True,"single_vs_batched":True,"common_pool_score_parity":True,
            "native_margin_max_absolute_error":native, "optional_pair_checked":pair_checked, "optional_A_C_checked":ac_checked,
            "no_external_socket":True,"top5_by_query":batched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path, help="Inference ZIP, or an extracted source directory for --code-only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--structural-only", action="store_true")
    mode.add_argument("--code-only",action="store_true")
    parser.add_argument("--image-dir",type=Path)
    parser.add_argument("--prompts",type=Path)
    parser.add_argument("--report",type=Path)
    args = parser.parse_args()
    if args.code_only:
        code_check((args.package/"model.py").read_bytes())
        result = {"validated_stage":"source_only", "requirements":check_pins((args.package/"requirements.txt").read_bytes())}
    else:
        with tempfile.TemporaryDirectory(prefix="haikuthon-check-") as tmp:
            work = Path(tmp)
            root = work/"extracted"
            result = inspect_zip(args.package,None if args.structural_only else root)
            if not args.structural_only:
                for name, pin in result["requirements"].items():
                    require(metadata.version(name) == pin, f"Install the ZIP pin {name}=={pin} before full QA")
                result.update(smoke(root,args.image_dir,args.prompts,work))
                result["installed_pins_match"] = True
    software = {"python":platform.python_version(),"system":platform.system()}
    for name in ("torch","torchvision","transformers","safetensors","pillow","tokenizers","huggingface-hub"):
        try:
            software[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    result.update(software=software,official_install_limits_verified=False,official_full_predict_verified=False)
    text = json.dumps(result,ensure_ascii=False,indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(text+"\n", encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
