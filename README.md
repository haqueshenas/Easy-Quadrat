<div align="center">

<img src="assets/EQ.png" alt="Easy Quadrat logo" width="190">

# Easy Quadrat

**The traditional quadrat, brought into image-based crop phenotyping.**

Version 2.0.0 · Free and open-source crop-phenotyping software

[![DOI](https://zenodo.org/badge/749297957.svg)](https://doi.org/10.5281/zenodo.23557475)

[Download Windows application (v2.0.0)](https://github.com/haqueshenas/Easy-Quadrat/releases/tag/v2.0.0) · [PDF user guide](docs/EasyQuadrat-v2.0.0-User-Guide.pdf) · [Source code (v2.0.0)](https://github.com/haqueshenas/Easy-Quadrat/tree/v2.0.0) · [Easy-Phenotyping Lab (EPL)](https://haqueshenas.github.io/EPL/)

*Select, crop, and segment quadrat-based sampling regions in crop images—with manual control, classical image processing, or deep learning.*

</div>

---

## Contents

- [1. Overview](#1-overview)
- [2. Quick start](#2-quick-start)
- [3. Understanding the outputs](#3-understanding-the-outputs)
- [4. Choosing a method](#4-choosing-a-method)
- [5. Method guide](#5-method-guide)
  - [5.1 Manual](#51-manual)
  - [5.2 Fixed Frame](#52-fixed-frame)
  - [5.3 Auto · Classic I](#53-auto--classic-i)
  - [5.4 Auto · Classic II](#54-auto--classic-ii)
  - [5.5 Auto · Deep Learning](#55-auto--deep-learning)
- [6. Image integrity and output policies](#6-image-integrity-and-output-policies)
- [7. Practical workflow and limitations](#7-practical-workflow-and-limitations)
- [8. What is new in v2.0.0?](#8-what-is-new-in-v200)
- [9. License and third-party components](#9-license-and-third-party-components)
- [10. Credits and contact](#10-credits-and-contact)

## 1. Overview

Easy Quadrat is nothing but the same quadrat traditionally used in agronomy and crop science. Now, it has been brought into the field of modern phenotyping, for cropping and sampling—not from the plants themselves, but rather from images of plants.

In crop phenotyping, the entire image of an experimental plot is not always needed. Researchers may need to extract only one or more specific sampling regions, remove unwanted margins, or isolate a target area before performing subsequent image analysis.

**White quadrats:** The three Auto methods are designed primarily
for images containing a visible white sampling quadrat. They are
not general-purpose detectors for quadrats of arbitrary colours.
Detection quality depends on contrast, lighting, quadrat size,
image resolution, and other image-acquisition conditions.

Easy Quadrat provides five methods for defining and extracting these regions from plant images:

1. **Manual** — draw a quadrat or polygon directly on an image.
2. **Fixed Frame** — apply the same proportional rectangular frame to a collection of images.
3. **Auto · Classic I** — identify a white quadrat using classical image-processing operations.
4. **Auto · Classic II** — use an alternative classical segmentation procedure.
5. **Auto · Deep Learning** — detect and segment quadrats using trained YOLO instance-segmentation models.

The methods serve different purposes. A researcher may use one method alone or combine methods in sequence, depending on the imagery and the objectives of the study. For example, an image cropped automatically may still contain unwanted margins. Its cropped output can then be passed to Fixed Frame for further trimming.

**The workflow is deliberately flexible:** Easy Quadrat provides the processing tools; the researcher decides which regions to extract, which outputs to use, and how the resulting images fit into the wider phenotyping workflow.

Easy Quadrat identifies and extracts image regions. It does not, by itself, measure or validate phenotypic traits.

## 2. Quick start

### Windows application

The easiest way to use Easy Quadrat is the ready-to-run Windows x64 application.

1. Download `EasyQuadrat-v2.0.0-Windows-x64.zip` from the [GitHub Releases page](https://github.com/haqueshenas/Easy-Quadrat/releases/latest).
2. Extract the complete ZIP archive to a folder on your computer.
3. Keep the `EasyQuadrat` folder and its `_internal` directory together. Do not move or run the executable separately from the files required by the application.
4. Run `EasyQuadrat/EasyQuadrat.exe`.
5. Choose one of the five methods from the main window.
6. Select the input image folder and an existing output folder in the chosen method.
7. Adjust the available settings, inspect the preview where provided, and run the desired operation.
8. Review the resulting images and any log or warning messages before using the outputs for scientific analysis.

The release archive includes sample images for initial testing. For an offline reference, see the [PDF user guide](docs/EasyQuadrat-v2.0.0-User-Guide.pdf).

### Important folder notes

- Select an input folder containing the images you want to process. The application does not recursively search nested subfolders; when working with previous outputs, select the specific folder containing the intended images.
- The selected output folder must already exist. Easy Quadrat creates the method-specific subfolders it needs.
- Existing output folders are reused; their contents are not automatically cleared. For a new processing run, use a separate, preferably empty, output folder to avoid mixing new results with earlier ones.

## 3. Understanding the outputs

Easy Quadrat distinguishes between **Cropped images** and **Segmented images**. They are different representations of the selected sampling region and are intended for different downstream tasks.

| Output | What it contains | When it is useful |
|---|---|---|
| **Cropped images** | A rectangular crop enclosing the selected or detected quadrat region. | When a smaller image containing the sampling region is needed for subsequent analysis. |
| **Segmented images** | An image retaining the source canvas dimensions, with pixels outside the selected or detected region replaced by the chosen background colour. | When the region needs to be visually isolated while retaining the surrounding image coordinates and canvas. |
| **Labels** (Manual) | Polygon annotations in COCO JSON and YOLO polygon-text formats. | When the selected regions are also needed as annotations for compatible modelling workflows. |
| **Log files** | Processing information, settings, results, warnings, and errors, where provided by the method. | For documenting and reviewing a processing run. |

Files in the cropped-image output are conventionally named with a `C_` prefix; segmented images use an `S_` prefix. Fixed Frame writes its cropped images directly into the selected output directory, while the other methods use method-specific subfolders.

A rectangular crop is not the same as a polygon-shaped extraction. If a quadrat is tilted in the original image, the rectangular crop may contain unwanted pixels near its corners. The corresponding segmented image can help exclude such pixels by replacing everything outside the selected or detected mask with the specified background colour.

The appropriate output depends on the objective of the subsequent analysis. Neither output automatically corrects a tilted quadrat or guarantees that the extracted region is suitable for every phenotyping task.

## 4. Choosing a method

| Method | Best suited to | Main principle |
|---|---|---|
| **Manual** | Images requiring direct human judgement, irregular regions, or precise polygon selection. | The user marks the sampling region on the image. |
| **Fixed Frame** | Consistently acquired images in which the desired region occupies approximately the same proportional position. | The same percentage-based rectangular frame is applied to each image. |
| **Auto · Classic I** | Images in which classical thresholding and morphological operations can distinguish the white quadrat. | Saturation thresholding and isotropic morphological operations. |
| **Auto · Classic II** | Similar tasks where directional morphological operations and convex-hull-based selection are useful. | Classical segmentation with horizontal and vertical directional processing. |
| **Auto · Deep Learning** | Images suited to trained quadrat-detection and instance-segmentation models. | YOLO predicts quadrat instances and their masks. |

Automatic detection is sensitive to contrast, lighting, quadrat
size, image resolution, and acquisition conditions. Inspect
representative results rather than assuming every detected
region is correct.

## 5. Method guide

### 5.1 Manual

Manual is the most direct way to define the sampling region when automatic detection is unsuitable or when the researcher wants explicit control over the selection.

Choose an input folder, choose an output folder, and select points on the displayed image to define the region. After confirming the selection with **Crop**, Easy Quadrat saves the resulting crop, segmented image, and polygon annotation files. It then advances to the next image.

#### Selection options

**Points**

- `2-Sq.` (default): click two opposite vertices defining the diagonal of a square. The two clicks represent opposite corners, not the endpoints of a side.
- `3`–`20`: select the specified number of polygon vertices.
- `Custom`: specify between 3 and 1,000 vertices for a custom polygon.

For a rectangular or polygonal sampling region that does not fit the square mode, select an appropriate vertex count or use `Custom`.

**Line**

Controls the display colour of the selection points and lines. It changes the selection overlay, not the colour of the source image.

**Background**

Sets the colour used outside the selected polygon in the segmented output. Available colours are Red, Green, Blue, Yellow, Magenta, Purple, Black, and White. The default is Black.

**EXIF Orientation**

Controls whether the image's EXIF Orientation metadata is applied before display and selection. It is off by default. Enable it when the orientation recorded in the image metadata should determine how the image is displayed and processed. The source image file itself is not modified.

**Zoom**

The slider controls magnification from 1× to 10×. You can also hold `Ctrl` and use the mouse wheel over the image. Right-drag to pan when zoomed in.

#### Buttons and keyboard shortcuts

| Control | Action |
|---|---|
| **Reset** / `Esc` | Clear the current selection and start selecting again. |
| **Crop** / `Enter` | Confirm the selection, save its outputs and annotations, and advance to the next image. |
| **Skip** / `Space` | Skip the current image and continue. |
| **Jump** | Move directly to a specified image number in the input list. |

Manual is particularly useful when the quadrat is partially obscured, the target shape is irregular, or the researcher needs to inspect and determine the sampling boundary personally.

### 5.2 Fixed Frame

Fixed Frame is a proportional trimming tool. Instead of detecting a quadrat, it applies a rectangular frame selected by the user to the images in the input folder.

It is useful when images have been acquired in a consistent manner—for example, when ground-based images of experimental plots contain unwanted borders, soil margins, neighbouring plots, or other areas that should be excluded before subsequent analysis.

#### Frame controls

The interface provides a preview, a preview-image selector, four percentage controls, and an orientation selector.

| Control | Meaning |
|---|---|
| **Left** | Horizontal position of the left frame boundary, expressed as a percentage of image width. |
| **Right** | Horizontal position of the right frame boundary, expressed as a percentage of image width. |
| **Top** | Vertical position of the top frame boundary, expressed as a percentage of image height. |
| **Bottom** | Vertical position of the bottom frame boundary, expressed as a percentage of image height. |
| **Reset frame** | Restore the full-frame boundaries. |
| **Preview image** | Choose an image from the input collection to inspect the selected frame. |
| **Initial orientation** | Choose Original, Horizontal, or Vertical. |

The default frame uses the complete image: Left = 0%, Top = 0%, Right = 100%, and Bottom = 100%.

The percentage values define the frame boundaries, not the percentage of pixels to remove from each side. The same percentage-based frame is applied to every image in the collection. Images with different dimensions therefore receive proportional—not necessarily identical pixel-sized—crops.

#### Orientation

- **Original:** do not request an additional horizontal or vertical orientation change.
- **Horizontal:** make the image landscape-oriented where necessary.
- **Vertical:** make the image portrait-oriented where necessary.

Horizontal and Vertical use a right-angle transpose when needed. This is not a deskewing operation: the program does not estimate a quadrat's tilt or rotate it to make its borders horizontal. The requested right-angle reorientation does not resize or interpolate the image pixels.

When the full frame and Original orientation require no image transformation, the output system can preserve the original file through an exact-byte copy.

Click **Crop images** to process the collection. Fixed Frame saves the derived cropped images in the selected output directory, using the `C_` filename prefix, and records run information in its log.

### 5.3 Auto · Classic I

Classic I uses classical image-processing operations to identify a white quadrat. It first identifies low-saturation candidate pixels and then refines the mask using morphological operations and connected-region processing.

The original Classic I approach was suggested by Peter Suter; see the [original discussion](https://discuss.python.org/t/how-can-i-detect-and-crop-the-rectangular-frame-in-the-image/32378/25).

#### Parameters

| Control | Default | Meaning |
|---|---:|---|
| **Saturation threshold (0–1)** | 0.1 | Pixels with HSV saturation below this threshold become candidates for the low-saturation mask. Adjust it when the quadrat and background are not being distinguished satisfactorily. |
| **Closing radius** | 40 px | Controls the scale of morphological closing, which can bridge small gaps and connect nearby mask regions. |
| **Opening radius** | 220 px | Controls the scale of the final morphological opening, which helps remove unwanted structures and refine the candidate region. |
| **Segmented background** | Black | Selects the colour used outside the final segmentation mask. |
| **Reset** | — | Restores the default parameter values. |

The useful parameter values depend on image resolution, the quadrat's dimensions, contrast and lighting. The defaults are starting points, not universally optimal settings.

#### Preview and processing

**Show preview** enables or disables the visual preview. Use **Run a test** to inspect the processing stages on the first input image without saving batch outputs. This makes it easier to understand how the mask develops and whether the current settings are suitable.

After inspecting the test, adjust the parameters if necessary and click **Run Classic I** to process the input collection.

The method generates cropped and segmented images and a `Log.txt` file documenting the batch processing results.

### 5.4 Auto · Classic II

Classic II provides an alternative classical approach. Like Classic I, it begins with saturation-based candidate pixels, but its subsequent processing includes directional morphological operations and convex-hull construction. The original candidate-area selection rule is retained in v2.0.0.

#### Parameters

| Control | Default | Meaning |
|---|---:|---|
| **Saturation threshold (0–1)** | 0.1 | Defines the low-saturation candidate mask. |
| **Closing radius** | 40 px | Controls the scale of morphological closing. |
| **Opening factor size** | 300 px | Controls the size of the rectangular footprint used for horizontal and vertical directional openings. Set it to zero to disable these directional openings. |
| **Segmented background** | Black | Selects the colour outside the final segmentation mask. |
| **Reset** | — | Restores the default parameter values. |

The optimal settings depend on the scale and appearance of the quadrat and on the characteristics of the image collection. A parameter setting that works well for one acquisition setup may not work equally well for another.

#### Preview and processing

As with Classic I, **Show preview** enables or disables the preview panel. **Run a test** examines the first input image using the current settings and does not save batch outputs. Review the processing stages before deciding whether the settings are suitable.

Click **Run Classic II** to process the image collection. Successful detections produce cropped and segmented outputs, while `Log.txt` records results, warnings, and errors.

### 5.5 Auto · Deep Learning

Auto · Deep Learning uses trained YOLO instance-segmentation models to detect quadrats and generate masks for accepted detections. The method offers four bundled models and supports compatible custom segmentation weights.

**First-use note:** When you first use Auto · Deep Learning after
installing the application, loading a built-in model may take
noticeably longer than other steps, depending on your computer.
The built-in model weights are included in the Windows package,
so they are normally loaded from local files rather than downloaded
from the internet. Please allow the model to finish loading before
assuming that the application has stalled. Some loading time may
recur when starting a separate test or batch run.

#### Model options

| Model | Description |
|---|---|
| **EQ1 — Standard (YOLOv8n-seg)** | The standard Easy Quadrat model based on YOLOv8n-seg. |
| **EQ2 — Standard (YOLOv8m-seg)** | A larger standard model based on YOLOv8m-seg. It may require more memory and processing time. |
| **EQ Frameless 1** | A model variant designed to reduce the influence of the quadrat frame. Results depend on the image and quadrat. |
| **EQ Frameless 2** | A second frame-reduced-influence model variant. Its performance also depends on the image and quadrat. |
| **Custom model (.pt)** | Load compatible YOLO instance-segmentation weights from a `.pt` file. |

A detection-only model is not sufficient: this method requires instance-segmentation weights that produce masks. Custom models must therefore be compatible with the YOLO instance-segmentation workflow.

EQ1 and EQ2 are not guaranteed to behave identically on every dataset. Test the models against representative images and inspect their detections before choosing a model for batch processing.

#### Detection parameters

| Control | Default | Meaning |
|---|---:|---|
| **Segmentation model** | EQ1 | Selects the built-in or custom model used for inference. |
| **Custom model file** | None | Specifies the compatible custom `.pt` model when Custom model is selected. |
| **Confidence threshold (0–1)** | 0.25 | Sets the minimum YOLO confidence threshold. Raising it makes acceptance stricter and may exclude valid quadrats; lowering it may accept less reliable detections. |
| **Maximum quadrats per image** | 1 | Sets the maximum number of quadrat instances the model may return for each input image. The default is 1; set a value greater than 1 when an image may contain multiple quadrats. This is an upper limit, not a guarantee that the model will detect that many quadrats. Actual detections depend on the model, image conditions, and confidence threshold. |
| **Segmented background** | Black | Sets the background colour outside the union of the accepted detection masks. |
| **Reset** | — | Restores the default model and detection parameters. |
| **Show preview** | Enabled | Shows or hides the visual test panel. Disabling it does not disable batch processing. |

#### Testing before batch processing

Use **Run a test** to inspect YOLO detections, the combined mask, and the final segmented preview for the first input image. The test does not save output files.

This preview is valuable because detection confidence alone cannot establish whether a quadrat has been extracted correctly. Inspect the proposed boundaries and mask, try another model or adjust the threshold if necessary, and assess the results in the context of the intended phenotyping task.

Click **Run Deep Learning** to process the input collection.

For accepted detections, the method generates one cropped image per detected quadrat and one segmented image per input image. When multiple quadrats are accepted in one image, their masks are combined in that image's segmented output. The batch also produces a `Log.txt` file.

## 6. Image integrity and output policies

Easy Quadrat v2.0.0 uses a centralized image-output system to make its saving behaviour more predictable and transparent across the processing methods.

The central output system plans filenames and formats, uses exact-byte copies for genuine no-operation cases, verifies saved pixels on supported lossless routes, and records relevant warnings when a requested saving strategy cannot be used as planned.

### Lossless output (recommended)

Lossless output is recommended when it is important to avoid introducing additional pixel changes during image encoding.

Where technically eligible and verifiable, JPEG coefficient-domain cropping can be used without re-encoding the JPEG pixels. When that route is unavailable or cannot be verified, the lossless policy may select an alternative format, such as PNG or TIFF, to preserve the processed raster.

A lossless output preserves the pixels of the processed result; it does not reverse the effects of cropping or segmentation, and it does not guarantee that every metadata field will be retained.

### Keep JPEG

The **Keep JPEG** option is intended for workflows that require JPEG output. Depending on the operation and JPEG block alignment, the program may need to re-encode the image. JPEG re-encoding is lossy, even at quality 100, so pixel values may change.

Use this option only when retaining JPEG format is more important than guaranteeing exact pixel preservation.

When the input collection contains JPEG or MPO images and an output-policy choice is required, the application presents the available policy before processing. Follow the on-screen information and review the warnings and log to understand which strategy was used.

### Image orientation and the quadrat's geometry

Easy Quadrat does not attempt to deskew a tilted quadrat, rectify perspective, or rotate the quadrat itself to make it geometrically horizontal or vertical. The rectangular crop remains aligned to the image axes.

Orientation metadata is a separate matter:

- In **Manual**, EXIF Orientation is optional and is disabled by default.
- In the Auto methods, EXIF Orientation is applied where present before detection.
- In **Fixed Frame**, EXIF orientation is interpreted, and the user may additionally request Horizontal or Vertical orientation where required.

These operations should not be confused with automatically straightening the quadrat. Fixed Frame's user-selected right-angle transpose does not resize or interpolate image pixels.

The application does not modify the original input files; it saves derived outputs separately. Some metadata, including EXIF or XMP fields, may not be copied to every derivative, especially when a format conversion or standalone image extraction is necessary. Where applicable, metadata and output warnings are recorded for review.

## 7. Practical workflow and limitations

### Combining methods

The five methods can be used independently or in sequence. For example:

1. Use Auto · Classic I, Classic II, or Deep Learning to identify and extract quadrats from a collection.
2. Inspect the resulting `Cropped images`.
3. If the rectangular crops still contain unwanted margins, choose that `Cropped images` folder as the input to Fixed Frame and trim the images further.
4. Use Manual when individual images need precise correction or an irregular sampling polygon.

This is only an example. The researcher determines the appropriate sequence according to the study design and image characteristics.

### Before using the results

- Inspect representative outputs, including images from different lighting conditions, acquisition dates, or plot appearances.
- Review warnings, errors, the selected output policy, and the processing log where available.
- Check whether the extracted region corresponds to the intended sampling area and whether residual margins are acceptable for the planned analysis.
- Use a fresh output directory for each run when you need to keep results separate and reproducible.

### Sampling tolerance and parameter choice

As in other areas of agricultural research, the appropriate
sampling precision depends on the study design, research
objectives, and the traits to be analysed. In large images
containing millions of pixels, a small number of residual edge
pixels or slight misalignment between a crop and the quadrat
boundary may be negligible for some plot-level analyses. However,
such errors may matter for small sampling areas, edge-sensitive
measurements, or pixel-level analyses.

There is no universally correct level of strictness for every
dataset. Choose the method and its settings according to the
research objective and image conditions. This may involve adjusting
the confidence threshold in Deep Learning, the saturation and
morphological parameters in the classical methods, the frame
boundaries in Fixed Frame, or the selected polygon in Manual.
Inspect representative outputs before applying a setting to an
entire dataset, and document choices that could affect comparisons
or conclusions.

### Quadrat frame and surrounding margins

Whether to retain visible parts of the white quadrat frame or surrounding margins depends on the objective of the subsequent analysis. The quadrat is a spatial reference, not vegetation. If its white-frame pixels are included in calculations of canopy cover, mean RGB values, greenness indices, or other pixel-based traits, they may affect the results and interfere with plant–background classification.

Likewise, pixels outside the intended sampling area may include soil, neighbouring plants, or other objects. Retaining or excluding these margins can therefore change the sampled area and the resulting statistics.

For analyses intended to characterize vegetation within the quadrat, exclude or mask frame pixels and out-of-quadrat margins from downstream calculations where appropriate. For other purposes, retaining some of this context may be useful for visual inspection, quality control, or a different analytical objective. The choice should be guided by the research question and applied consistently across comparable images.

Easy Quadrat defines and extracts quadrat-based image regions; it does not perform a separate, plant-only segmentation that guarantees removal of every frame pixel or every unwanted object within the selected or detected region. Inspect the outputs and determine whether further masking or processing is needed for the intended analysis.

Finally, ordinary RGB pixel values should not be interpreted as calibrated surface reflectance unless appropriate image-acquisition and calibration procedures have been applied.


The supported input formats depend on the selected method and the image formats available to its reader and writer components. Common formats include JPEG, PNG, TIFF, BMP and WebP; certain methods also handle MPO and GIF images.

Auto · Classic I, Auto · Classic II, and Auto · Deep Learning reject unsupported high-bit-depth inputs rather than silently reducing their pixel precision. Some other image modes may be normalized for processing, and metadata preservation can depend on the selected method and output format. Consult the log and warnings whenever an image is converted or a fallback saving strategy is used.

**Easy Quadrat extracts sampling regions; it does not establish their agronomic representativeness, validate experimental design, or replace downstream phenotypic measurements.** Those decisions remain the responsibility of the researcher.

## 8. What is new in v2.0.0?

Version 2.0.0 retains the five-method approach of Easy Quadrat v1.0 while restructuring the application and strengthening the user workflow.

- **Redesigned interface:** a more consistent main window and method-specific interfaces, with clearer controls and more room for image previews.
- **Interactive previews for Auto methods:** users can inspect processing stages or YOLO detections on a test image without saving batch outputs.
- **Centralized image-output handling:** filename and format planning, exact-byte copies for genuine no-operation cases, verified lossless saving where supported, and explicit warnings about potentially lossy JPEG re-encoding.
- **Clearer batch records:** the Auto methods record the input and output directories, processing parameters, results, warnings, and errors in their logs.
- **More modular source code:** the original application was concentrated in a single Python source file; v2.0.0 separates the methods, common image-processing and output functions, user-interface components, assets, and bundled resources into a more maintainable project structure.
- **A ready-to-run Windows package:** the release is distributed in PyInstaller One Directory format, with the required application files and sample images.

These changes focus on usability, reproducibility, maintainability, and safer image handling rather than claiming that the underlying segmentation algorithms are universally more accurate.

## 9. License and third-party components

The Easy Quadrat application source is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the license text.

The **Auto · Deep Learning** method uses the Ultralytics YOLO framework and trained model weights with separate third-party licensing terms. The project identifies the YOLO-related component and its bundled model files as subject to the **GNU Affero General Public License v3.0 (AGPL-3.0)**. The MIT license for the application's own source should not be interpreted as replacing those separate terms.

Other bundled components, including libjpeg-turbo, retain their applicable third-party license and notice requirements.

Before using or redistributing the relevant components, read [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt), the applicable license texts provided with the project, and the relevant upstream terms. Users are responsible for ensuring that their intended use and redistribution comply with the licenses that apply to each component.

## 10. Credits and contact

**Author:** Abbas Haghshenas <br>
**Project:** Easy Quadrat — Easy-Phenotyping Lab (EPL) <br>
**Email:** [haqueshenas@gmail.com](mailto:haqueshenas@gmail.com) <br>
**Website:** [Easy-Phenotyping Lab](https://haqueshenas.github.io/EPL/) <br>
**Repository:** [haqueshenas/Easy-Quadrat](https://github.com/haqueshenas/Easy-Quadrat)

For questions, reproducible bug reports, or suggestions, please use the repository's [GitHub Issues](https://github.com/haqueshenas/Easy-Quadrat/issues) page.

---

<div align="center">

**Easy Quadrat · Version 2.0.0**

Quadrat-based sampling for image-based crop phenotyping

[Easy-Phenotyping Lab (EPL)](https://haqueshenas.github.io/EPL/)

Copyright © 2026 Abbas Haghshenas · MIT License for the Easy Quadrat application source

</div>
