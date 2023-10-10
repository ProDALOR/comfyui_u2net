from .U2NetNodes import U2NetLoader, U2NetSegmentation, U2NetChToMask, U2NetToMask, U2NetBaseNormalization, U2NetMaxNormalization

NODE_CLASS_MAPPINGS = {
    "U2NetLoader": U2NetLoader,
    "U2NetSegmentation": U2NetSegmentation,
    "U2NetChToMask": U2NetChToMask,
    "U2NetToMask": U2NetToMask,
    "U2NetBaseNormalization": U2NetBaseNormalization,
    "U2NetMaxNormalization": U2NetMaxNormalization
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "U2NetLoader": "Load U2Net model",
    "U2NetSegmentation": "U2Net segmentation",
    "U2NetToMask": "To mask",
    "U2NetChToMask": "Segmentation to mask"
}