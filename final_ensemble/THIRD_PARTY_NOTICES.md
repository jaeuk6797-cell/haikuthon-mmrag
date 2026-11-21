# Pretrained components

The retrieval model incorporates Google's SigLIP base, patch size 16, trained at 224 pixels. This distribution merges the task-trained SigLIP heads into that model and ships the local processor and tokenizer. The model card identifies the Apache License 2.0. [Model card](https://huggingface.co/google/siglip-base-patch16-224) · [License](LICENSE-SigLIP.txt)

The image-text matching model incorporates Salesforce BLIP ITM base COCO and its local processor and tokenizer. This distribution includes the public model, separately selectable trained text and matching layers, and a task-specific two-image comparator. Copyright (c) 2022, Salesforce.com, Inc. The model card identifies the BSD 3-Clause License. [Model card](https://huggingface.co/Salesforce/blip-itm-base-coco) · [Upstream notice](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) · [License](LICENSE-BLIP.txt)
