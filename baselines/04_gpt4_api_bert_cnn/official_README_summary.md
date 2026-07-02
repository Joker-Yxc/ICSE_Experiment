# Official Repository README Notes

Source: `https://github.com/yan-scnu/Prompted_Dynamic_Detection`

The official repository README for **Prompt Engineering-assisted Malware Dynamic Analysis Using GPT-4** describes the method but says that the authors do not publish the generated GPT content, training weights, or full training code in the public repository. It also says researchers can contact the authors by email for code access.

Pipeline described in the official README:

1. Use API-call sequences as dynamic malware behavior features.
2. Ask GPT-4 to create explanatory text for each API call.
3. Use BERT to encode the explanatory text.
4. Use those vectors as API-call representations and form sequence embeddings.
5. Train a CNN detection model for malware detection.

This baseline adapter implements that public pipeline locally and caches generated API descriptions in `api_descriptions.json`.

