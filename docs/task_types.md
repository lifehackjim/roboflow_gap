# Task Types in Computer Vision

Computer vision lets computers understand pictures in various ways.

Some models can sort pictures by category. They can tell if a picture shows a dog, a cat, or something else.

Other models find and show where things are in a picture. They can point out exactly where in the picture a dog is sitting or a car is parked.

Then, there are models that do both jobs. They can spot all the cars in a parking lot and also say what kind they are.

In short, computer vision uses different models to make sense of pictures, from simple sorting to identifying and locating objects.

## classification

Acts like a smart sorting hat for images. A computer views an image and labels it with a category or multiple categories, such as "cats," "dogs," or "beaches," without pointing out where in the image these objects are.

### Examples

1. Binary Classification: Making a pick between two options, like deciding whether an image shows a cat or a dog.
2. Multi-Class Classification: Sorting an image into one of several buckets at once, whether it's a cat, a dog, a bird, etc.
3. Image Sorting: Like having an automatic helper that puts your images into albums titled "beach vacations" or "mountain trips."

### Pros

1. Easy to Set Up: Simpler and need less computing power.
2. Very Useful: Handy in a wide range of fields, from healthcare to enhancing smartphone capabilities.
3. Great for Simple Images: Shines when images have one main subject, like a singular cat or a lone tree.

### Cons

1. No Details on Location: Doesn't specify where items are within the image.
2. Limited Choices: Can only identify items it has been trained on.
3. Struggles with Busy Images: Has difficulty with images that have lots happening.

### Inference Models

- YOLOv8 (You Only Look Once version 8): Lightning fast at labeling images with the right category or categories.
- ViT (Vision Transformer): Breaks down images in a way that's similar to processing language, enhancing its ability to understand complex scenes.

### Inference Model Infer Arguments

```
image (Any): The image or list of images to be processed.
disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
return_image_dims (bool, optional): If set to True, the function will also return the dimensions of the image. Defaults to False.
```

## object-detection

Finds and labels different objects in an image, showing each with its own box and name tag.

### Examples

1. Automated Surveillance: Watches over places by spotting people, vehicles, or items in video feeds.
2. Self-Driving Cars: Aids vehicles in recognizing pedestrians, other cars, and street signs.
3. Retail Analytics: Tracks customer numbers and observes which products catch their interest.

### Pros

1. Spatial Context: Shows where objects are and how big they are in the image.
2. Multi-Object Identification: Spots and names several objects at once.
3. Broad Uses: Key for keeping an eye on areas, managing traffic, and studying customer behavior in shops.

### Cons

1. More Complex: Needs powerful computers and sophisticated software.
2. Time-Consuming Setup: Requires lots of manually marked images for training.
3. May Struggle with Speed: The most accurate models might not work fast enough for live video.

### Inference Models

- YOLO-NAS (You Only Look Once - Neural Architecture Search): Quick and sharp at finding various items.
- YOLOv8 (You Only Look Once version 8): Praised for swift and precise object detection.
- YOLOv5 (You Only Look Once version 5): Favored for its efficient blend of speed and accuracy.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
fix_batch_size (bool, optional): If True, fix the batch size for predictions. Useful when the model requires a fixed batch size. Defaults to False.
iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
return_image_dims (bool, optional): Whether to return the dimensions of the processed images along with the predictions. Defaults to False.
```

## instance-segmentation

Takes apart an image into its pieces, marking every object down to the exact pixel, so each one stands out on its own.

### Examples

1. Medical Imaging: Helps doctors see details in body scans, like spotting tumors or differentiating tissues.
2. Agriculture: Assists in counting crops from the air and checking their health, making farming more efficient.
3. Robotics: Lets robots 'see' objects clearly, including their shapes and where they are, to interact or pick them up correctly.

### Pros

1. Super Detailed: Draws exact lines around each object, even if they're touching or overlapping.
2. In-Depth Understanding: Gives a full breakdown of everything in the picture, making it clearer what's what.
3. Works in a Crowd: Really good at picking out individual items in busy pictures where lots of things are close together.

### Cons

1. Really Complex: Needs a lot of computer power to run because it's doing a very detailed job.
2. Needs a Lot of Prep: To learn this skill, it needs a ton of pictures where every little part has been marked out by hand.
3. Takes Its Time: Not the quickest, especially when compared to simpler tasks like just spotting objects or labeling them.

### Inference Models

- YOLACT (You Only Look At CoefficienTs): A trailblazer in cleanly separating objects from their background with sharp precision.
- YOLOv8 (You Only Look Once version 8): Not only finds but also intricately maps out every object in a scene.
- YOLOv7 (You Only Look Once version 7): Stands out for its precise ability to segment and outline various objects, showing every edge and curve.
- YOLOv5 (You Only Look Once version 5): Speedy at detecting objects and then goes further to detail them out.

### Inference Model Infer Arguments

```
image (Any): The image or list of images to segment. Can be in various formats (e.g., raw array, PIL image).
class_agnostic_nms (bool, optional): Whether to perform class-agnostic non-max suppression. Defaults to False.
confidence (float, optional): Confidence threshold for filtering weak detections. Defaults to 0.5.
disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
iou_threshold (float, optional): Intersection-over-union threshold for non-max suppression. Defaults to 0.5.
mask_decode_mode (str, optional): Decoding mode for masks. Choices are "accurate", "tradeoff", and "fast". Defaults to "accurate".
max_candidates (int, optional): Maximum number of candidate detections to consider. Defaults to 3000.
max_detections (int, optional): Maximum number of detections to return after non-max suppression. Defaults to 300.
return_image_dims (bool, optional): Whether to return the dimensions of the input image(s). Defaults to False.
tradeoff_factor (float, optional): Tradeoff factor used when `mask_decode_mode` is set to "tradeoff". Must be in [0.0, 1.0]. Defaults to 0.5.
```

## keypoint-detection

Zooms in on important spots on objects, like where limbs connect on a body or features on a face.

### Examples

1. Pose Estimation: Figures out body positions, great for fitness apps or analyzing sports movements.
2. Facial Recognition: Picks out individuals by mapping the unique landmarks of their faces.
3. Animation: Brings cartoon figures to life by copying how real people move.

### Pros

1. Detail-Oriented: Gets into the nitty-gritty of an object's structure by marking specific spots.
2. Flexible: Handy for a bunch of different uses, from tech that reacts to you to creating digital art.
3. Smarter Processing: Gets the job done without needing as much computer power as some other tasks.

### Cons

1. Needs a Helping Hand: Usually has to start with finding the object before it can map out the details.
2. Can Miss Things: Might not catch every detail if parts of the object are hidden.
3. Sticks to What It Knows: Only works on objects it's been taught to recognize.

### Inference Models

- YOLOv8 (You Only Look Once version 8): Really shines at pinpointing these crucial spots, helping to understand how someone is posing or expressing themselves.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
fix_batch_size (bool, optional): If True, fix the batch size for predictions. Useful when the model requires a fixed batch size. Defaults to False.
max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
return_image_dims (bool, optional): Whether to return the dimensions of the processed images along with the predictions. Defaults to False.
disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
```

## embedding

Turns images into a special format that lets computers figure out which ones look alike, by changing them into a set of numbers that represent the image's content.

### Examples

1. Recommendation Systems: Suggests items you might like, similar to ones you've viewed before.
2. Photo Organization: Helps sort your digital photo collection into groups by what's in the pictures.
3. Anomaly Detection: Identifies items or features that stand out from the norm, useful in quality control.

### Pros

1. Versatile: Useful for a range of applications, from online shopping to managing large image collections.
2. Efficient Sorting: Groups images by similarity, making it easier to manage and understand large datasets.
3. Data Compression: Reduces the size of image data, making storage and processing more efficient.

### Cons

1. Complexity: The way it determines similarity between images can be complex and not always transparent.
2. Data Quality Dependence: Its effectiveness is directly related to the quality and variety of the training images.
3. Potential for Bias: Might reinforce existing biases in the training data, affecting diversity and discovery of new content.

### Inference Models

- CLIP (Contrastive Language-Image Pretraining): A model that uses text descriptions alongside images to better understand and categorize images, making it highly effective at recognizing and grouping images based on their content.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
```

## gaze-detection

Tracks where someone is looking, giving clues about what they find interesting or important.

### Examples

1. Usability Testing: Helps designers see what parts of a website or app people look at most.
2. Assistive Technology: Allows people with limited mobility to use computers or devices just by looking at them.
3. Attention Analysis: Studies which parts of an advertisement or educational content catch people's attention.

### Pros

1. Improves Interaction: Makes technology easier and more natural to use by following our gaze.
2. Offers Insights: Provides valuable information about where people focus their attention.
3. Increases Accessibility: Helps people with disabilities interact with technology using their eyes.

### Cons

1. Sensitive to Conditions: Works best under certain lighting conditions and can be thrown off by changes.
2. Requires Special Equipment: Accurate gaze tracking often needs extra gadgets, which can be costly.
3. Raises Privacy Issues: Keeping track of where someone is looking could lead to concerns about privacy.

### Inference Models

- Gaze: Focuses on identifying the direction of someone's gaze to improve user experience and gather data on viewer engagement.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
```

## lmm

Turns images into words, explaining or answering questions about what's seen.

### Examples

1. Image Captioning: Comes up with a description for a photo.
2. Visual Question Answering: Answers specific questions about an image, like "Who is in the photo?"
3. Visual Commonsense Reasoning: Makes educated guesses about things not directly visible in the image, such as "What's likely to happen next?"

### Pros

1. Adds Depth: Gives a richer perspective on what's happening in an image beyond what's visible.
2. Enhances Interaction: Bridges the gap between human conversation and computer vision.
3. Broad Applications: Useful in areas like accessibility tech, content creation, and AI interactions.

### Cons

1. High Demand: Requires significant computational resources and extensive data to function.
2. Data Hungry: Needs a large collection of images paired with descriptive text to train effectively.
3. Sometimes Vague: The explanations or answers it provides may sometimes be ambiguous or not fully accurate.

### Inference Models

- CogVLM (Cognitive Vision and Language Model): A model skilled in describing images and answering questions by leveraging knowledge of both vision and language.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
```

## ocr

Turns pictures with writing into text you can use.

### Examples

1. Document Digitization: Makes editable text out of paper documents.
2. License Plate Recognition: Identifies car plates automatically for parking or toll systems.
3. Business Card Reading: Digitizes contact details from business cards quickly.

### Pros

1. Easy to Search: Turns text in images into something you can search and analyze.
2. Saves Time: Automates typing out documents or entering data by hand.
3. Handy: Converts hard-to-use printed or handwritten notes into digital format.

### Cons

1. Not Always Perfect: Might get confused by different fonts or messy handwriting.
2. Limited by Language: Best with common languages and might struggle with rare or complex scripts.
3. Misses the Big Picture: Can read words but doesn't always get what they mean together.

### Inference Models

- DocTR (Document Text Recognition): Like a super reader, it quickly turns written words in images into digital text you can edit and search.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
```

## unsupervised-segmentation 

Automatically separates things in an image, figuring out what's what all on its own.

### Examples

1. Scene Decomposition: Breaks down busy scenes into understandable parts, like cars from trees.
2. Object Discovery: Finds and recognizes things in pictures without needing to know what they are ahead of time.
3. Background Separation: Distinguishes the main subject from everything else in the background.

### Pros

1. No Pre-tagged Pictures Needed: Works without needing pictures to be labeled in advance.
2. Finds New Things: Can spot and identify objects that haven't been specifically taught.
3. Works on Many Types of Pictures: Good at dealing with a wide range of images without needing special adjustments.

### Cons

1. Results Can Vary: Sometimes, it might not separate things in the way you expect.
2. Needs Further Analysis: Just splitting up an image doesn't always tell you what the parts are.
3. Performance Fluctuates: How well it works can change a lot depending on the picture.

### Inference Models

- SegmentAnything: This model is like a curious explorer, looking at pictures and figuring out on its own what's in them, which is great for sorting through lots of images or finding interesting patterns.

### Inference Model Infer Arguments

```
image (Any): The input image or a list of images to process.
```
