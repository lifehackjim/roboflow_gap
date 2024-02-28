# roboflow_gap: Simplifying Computer Vision for Everyone

Fill in the GAP: gather, analyze, and process images and videos in seconds — no coding required.

**Aim**: To make computer vision effortless and accessible, enabling everyone to harness the power of advanced image analysis and processing without needing deep coding knowledge.

## Inspiration

Drawing inspiration from the user-friendly approaches of [Ollama](https://ollama.com/) and [AnythingLLM](https://useanything.com/), our mission is to democratize advanced image processing and analysis with [Roboflow](https://roboflow.com/), making it accessible to all.

## Introduction

The computer vision workflow involves three key stages: gathering, analyzing, and processing images. 

Creating tools for these stages typically demands considerable time, domain expertise, and coding skills.

Enter roboflow_gap. Designed to streamline your computer vision projects, it will eventually offer:

- **Gather**: An intuitive interface to collect images from cameras, screenshots, the internet, and more.
- **Analyze**: Straightforward access to Roboflow's comprehensive computer vision libraries for detailed image analysis and detection.
- **Process**: Handy tools for adding annotations to images, setting the stage for further insights and connections, such as linking identified objects to their Wikipedia entries.
- **CLI Tool**: An easy-to-use command-line interface that brings the power of roboflow_gap to all users, regardless of their technical skills.
- **User Support**: Ensuring that even complex computer vision tasks are within reach, with straightforward instructions and supportive feedback for users at any technical level.
- **TBD**:
  - **WebUI**: A visual interface to monitor the progress and outcomes of your roboflow_gap workflows, making it even easier to see the results of your computer vision projects in real time.
  - **Benchmarks**: Ability to benchmark different models and tasks to find the best fit for your use case.

## Analyzer Task Types

Supported by the [inference python package](https://github.com/roboflow/inference/), a broad spectrum of task types awaits, each tailored for specific capabilities and use cases.

Dive into our [Task Types documentation](docs/task_types.md) for an in-depth exploration of each task type.

Quick highlights include:

- classification: Easily labels images with categories but doesn't show where objects are.
- object-detection: Finds and shows where objects are in an image.
- instance-segmentation: Marks the exact edges of each object in an image, showing every detail.
- keypoint-detection: Points out important spots on objects, like where parts of the body are.
- embedding: Changes images so that similar ones can be grouped together, helping with searching and sorting.
- gaze-detection: Figures out where someone is looking, helping us understand what catches their eye.
- lmm (Language Models for Machines): Combines seeing images and understanding words to describe what's in the images or answer questions about them.
- ocr (Optical Character Recognition): Turns text in images into text you can edit and search.
- unsupervised-segmentation: Finds and separates objects in images all by itself, without needing examples to learn from.

## Key Features:

1. **Seamless Configuration Across Platforms**:
   - Utilize Pydantic for centralized configuration, harmonizing settings across the API, Web UI, CLI, and Python library.
   - Configuration settings are automatically loaded from environment variables for ease of use.

2. **Versatile Image Gathering**:
   - Efficiently collect images from a variety of sources, including cameras, screenshots, and files, using asynchronous techniques for real-time monitoring.

3. **Dynamic Image Analysis**:
   - Tailor analysis to specific computer vision tasks such as classification and object detection, with unique configurations for each, facilitated by Pydantic models.

4. **Customizable Image Processing**:
   - Apply a range of annotators to processed images, each with bespoke options defined in Pydantic models, allowing for dynamic annotation pipelines.

5. **Enrichment and Advanced Callbacks**:
   - Post-analysis, enrich image data with asynchronous callbacks for tasks like reverse image search and Wikipedia lookups.

6. **Flexible Output Options**:
   - Configure outputs to suit user needs, from displaying on the desktop to saving in a directory, complemented by comprehensive formatting options.

7. **Accessible Interfaces**:
   - Develop a user-friendly Web UI and a comprehensive CLI, making advanced computer vision tasks accessible to users of varying technical backgrounds.

8. **Comprehensive Documentation and Testing**:
   - Ensure reliability and performance with extensive documentation and testing, using tools like pytest for thorough validation.

## Project Objectives:

- **Democratize Computer Vision**: Simplify access to computer vision, enabling users from non-technical backgrounds to perform complex analyses with ease.
- **Support Extensive Vision Tasks**: From basic categorization to detailed segmentation, cater to a broad spectrum of computer vision applications through a modular design.
- **Offer Diverse Interaction Modes**: Whether through an API, a Web UI, or a CLI, provide varied interfaces to accommodate different user preferences.
- **Streamline Configuration and Customization**: Enable easy setup and customization of computer vision tasks, offering sensible defaults and flexible overrides for advanced users.
- **Facilitate Comprehensive Image Processing**: Beyond analysis, enrich and annotate images with a suite of tools for detailed image enhancement and interpretation.

By meeting these objectives, roboflow_gap intends to bridge the technical divide, bringing sophisticated computer vision capabilities within reach of a broad audience, fostering innovation and creativity across various applications.

## CLI Design

Ideally, we want anyone to be able to run the CLI with only a command group and no arguments. 

```bash
roboflow_gap clipboard
roboflow_gap paths
roboflow_gap cameras
roboflow_gap urls
roboflow_gap screenshots
```

It would automatically use environment variables for configuration and prompt the user for required values if they are not set or are not valid.

Every thing that can be made optional should be made optional, and the user should be able to override any setting with an argument.

### common-args

Common arguments for all command line interfaces.
Used to control models and processors.

```bash
--api-key {str} 
   required
   env: ROBOFLOW_API_KEY
--model {str}[:{key}={value}]
    env: ROBOFLOW_MODELS
    optional, multiple, default: classification
    optional argument key/value pairs dependent on model task_type
    enum: classification, object-detection, instance-segmentation, keypoint-detection, embedding, 
          gaze-detection, lmm, ocr, unsupervised-segmentation
    help: each image gathered will be fed to every model specified
    examples:
    --model instance-segmentation
    --model object-detection:confidence=0.5
    --model face-detection-mik1i/18:confidence=0.5
    --model yolov8x-pose:confidence=0.5 
--process {str}[:{key}={value}] 
    env: ROBOFLOW_PROCESSORS
    optional, multiple, default: label, boundingbox
    optional argument key/value pairs dependent on processor
    enum: polygon, boundingbox, orientedbox, mask, color, halo, ellipse, boxcorner, circle, dot, label, 
          blur, trace, heatmap, pixelate, triangle, roundbox, percentagebar, more TBD
    help: each image gathered will be processed by every processor specified
    examples:
    --process polygon:thickness=2 
    --process label
    --process wiki_lookup:language=en 
    --process reverse_image_search
```

### settings-args

Settings arguments for all command line interfaces.

Controls logging and verbosity and env file loading.

```bash
--dotenv {path}
    env: ROBOFLOW_DOTENV
    optional, default: .env
--quiet/--no-quiet 
    env: ROBOFLOW_QUIET
    optional, default: False
--log-console/--no-log-console 
    env: ROBOFLOW_LOG_CONSOLE
    optional, default: False
--log-console-format {str} 
    env: ROBOFLOW_LOG_CONSOLE_FORMAT
    optional, default: ...
--log-console-level {level} 
    env: ROBOFLOW_LOG_CONSOLE_LEVEL
    optional, default: info
    enum: debug, info, warning, error, critical
--log-file/--no-log-file 
    env: ROBOFLOW_LOG_FILE
    optional, default: True
--log-file-level {level} 
    env: ROBOFLOW_LOG_FILE_LEVEL
    optional, default: info
    enum: debug, info, warning, error, critical
--log-file-format {str} 
    env: ROBOFLOW_LOG_FILE_FORMAT
    optional, default: ...
--log-file-rotate/--no-log-file-rotate 
    env: ROBOFLOW_LOG_FILE_ROTATE
    optional, default: True
--log-file-name {file} 
    env: ROBOFLOW_LOG_FILE_NAME
    optional, default: roboflow_gap.log
--log-file-path {path} 
    env: ROBOFLOW_LOG_FILE_PATH
    optional, default: .
--log-file-max-mb {int} 
    env: ROBOFLOW_LOG_FILE_MAX_MB
    optional, default: 10
--log-file-max-files {int} 
    env: ROBOFLOW_LOG_FILE_MAX_FILES
    optional, default: 5
--log-logger-level {logger}={level} 
    env: ROBOFLOW_LOG_LOGGER_LEVEL
    optional, default: None
    examples:
    --log-logger-level inference=warning
```

#### watch-args

Watch arguments for all command line interfaces that have a watch concept.

```bash
--watch/--no-watch
    env: ROBOFLOW_WATCH
    optional, default: False
--sleep {int}
    env: ROBOFLOW_SLEEP
    optional, default: 0.5
--max-images {int}
    env: ROBOFLOW_MAX_IMAGES
    optional, default: None
--max-seconds {int}
    env: ROBOFLOW_MAX_SECONDS
    optional, default: None
```

### command group: clipboard

Grab images or videos from the clipboard for analysis. 

Can watch the clipboard and process them as they appear.

Example:

```bash
roboflow_gap clipboard --include-text --api-key YOUR_API_KEY
```

Arguments:

```bash
--include-text/--no-include-text
    # To include or not include text from clipboard in inference.
    env: ROBOFLOW_INCLUDE_TEXT
    optional, default: False
**watch-args
**common-args
**settings-args
```

### command group: paths

Grab images or videos from specified files or dirs for analysis.

Can watch files for changes in modification time and directories for new/changed files and process them as they appear.

Examples:
```bash
roboflow_gap paths --path "/path/to/your/image.jpg" --path "/path/to/dir" --recurse --api-key YOUR_API_KEY
```

Arguments:
```bash
--path {path} 
    env: ROBOFLOW_PATHS
    optional, default: [.]
--recurse/--no-recurse 
    env: ROBOFLOW_RECURSE
    optional, default: True
--wildcards {str} 
    env: ROBOFLOW_WILDCARDS
    optional, multiple, default: *.gif,*.png,...
    examples:
    --wildcards *.gif,*.png,*.jpg,*.jpeg,*.bmp,*.dib,*.jp2,*.jpe,..,
    --wildcards *.gif --wildcard *.png --wildcard *.jpg ...
--wildcard-append/--no-wildcard-append 
    env: ROBOFLOW_WILDCARD_APPEND
    optional, default: True
    help: append supplied wildcards to default wildcards
**watch-args
**common-args
**settings-args
```

### command group: cameras

Grab images from specified cameras for analysis.

Can watch cameras and process them as they appear.

Examples:
```bash
roboflow_gap cameras --api-key ABBB
```

Arguments:
```bash
--camera {int} 
    env: ROBOFLOW_CAMERAS
    optional, multiple, default: [0]
    help: should allow user to list available cameras by int index
**watch-args
**common-args
**settings-args
```

### command group: urls

Grab images or videos from specified urls for analysis.

Can watch rtsp, http, & https urls and process them if they change.

Examples:
```bash
roboflow_gap urls --url {url} --proxy {url} --api-key ABBB
```

Arguments:
```bash
--url {url} 
    env: ROBOFLOW_URLS
    required, multiple
    help: url to image
--proxy {str} 
    env: ROBOFLOW_PROXY
    optional, default: None
    help: proxy for url
**watch-args
**common-args
**settings-args
```

### command group: screenshots

Grab images from your desktop for analysis.

Can watch the desktop and process them as they appear.

Examples:
```bash
robogap screenshots --watch --sleep 0.5 --max-images 10 --max-seconds 60 --api-key ABBB
```

Arguments:
```bash
--monitor {int}
    env: ROBOFLOW_MONITORS
    optional, multiple, default: []
    help: should allow user to list available monitors by int index
--windowid {?}
    env: ROBOFLOW_WINDOWIDS
    optional, multiple, default: None
    help: should allow user to list available windowids (?)
--dimensions {int},{int}
    env: ROBOFLOW_DIMENSIONS
    optional, multiple, default: None
    help: should allow user to list available dimensions (?)
**watch-args
**common-args
**settings-args
```

