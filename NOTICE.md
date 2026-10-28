# Attribution and licensing notes

## Project material

The repository has not selected a project-wide open-source license. Source code, numerical outputs, and research text should not be assumed to inherit a dependency or dataset license. The original course assignment and supplied task instructions are excluded.

## Public datasets

- **Online News Popularity:** Fernandes, Vinagre, Cortez, and Sernadela (2015), UCI Machine Learning Repository, DOI [10.24432/C5NS3V](https://doi.org/10.24432/C5NS3V). The recorded UCI source identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **UJIIndoorLoc:** UCI Machine Learning Repository, DOI [10.24432/C5MS59](https://doi.org/10.24432/C5MS59). Author attribution is retained in the frozen source manifest; the recorded UCI source identifies CC BY 4.0.
- **Leukemia feature matrix:** the [LIBSVM binary dataset collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia). Redistribution permission for the checked dataset snapshot was not clearly established; no raw, merged, or standardized input matrix is included.

Data attribution is separate from the research project's authorship. Source manifests and download code are retained; original data are obtained explicitly from their official sources.

## Fonts embedded in report PDFs

The Chinese reports use the Droid Sans Fallback font supplied through PyMuPDF/MuPDF. The generated standalone TTF is excluded from this repository. Embedded subsets in the PDFs retain the third-party font's attribution:

> Copyright (c) 2005-2008, The Android Open Source Project

The font is distributed under the Apache License, Version 2.0. The applicable upstream [Droid font NOTICE](https://github.com/ArtifexSoftware/mupdf/blob/master/resources/fonts/droid/NOTICE) and [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) are the authority for its terms. Copies are provided in `third_party/droid_fonts/`.

## Dependencies

NumPy, SciPy, scikit-learn, pandas, Matplotlib, threadpoolctl, skglm, Numba, llvmlite, ReportLab, PyMuPDF, and the other declared dependencies are installed separately. Their own licenses apply. No dependency installation tree or compiled binary is vendored here.
