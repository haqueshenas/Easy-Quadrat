# Easy Quadrat v 1.0


  
Easy Quadrat is nothing but the same quadrat traditionally used in agronomy and crop science. Now, it has been brought into the field of modern phenotyping, for cropping and sampling; not from the plants themselves but rather from the images of plants.

**Overview**

The initial step in RGB image analysis for phenotyping involves segmenting the intended canopy area from the background. Although commercial phenotyping platforms usually employ built-in segmentation algorithms, researchers may opt to supervise the image segmentation process. This can be a time-consuming task, especially when dealing with numerous experimental plots. Easy Quadrat, a Python-based freeware, addresses this challenge by providing a set of simple tools to speed up the determination, segmentation, and cropping of sampling areas in ground-based images captured from crop canopies.
Depending on the study's goals and conditions, the segmentation task can be performed manually or automatically (the latter of which requires a white traditional quadrat to be present in the image). Even in the manual method, efforts are made to streamline and speed up the selection and segmentation processes. Easy Quadrat offers five image segmentation methods, known as Cropping methods. The three automated methods are designed to find and segment a white sampling quadrat in the image. For detailed information on each method, refer to the relevant section in the README.pdf file (https://github.com/haqueshenas/Easy-Quadrat/blob/main/README.pdf).

**Getting Started**

1.	Download and unpack the Easy Quadrat package
2.	Run EasyQuadrat.exe to open the main window.
3.	Specify the input path (original images from the canopy) and output path (desired location for saving outputs).
4.	Choose the cropping method.
5.	Click the Ok button.

Note: Ensure that the output path is an empty directory to prevent potential unexpected errors.

EQ Frameless models: If you wish to have a try to mitigate or eliminate the impact of quadrat frames, download the EQ_Frameless_1 or EQ_Frameless_2 models from the YOLOPlusModels directory and run them as custom models (please see README.pdf). These YOLOv8n-seg-type models have been trained almost using the same dataset utilized for training EQ1 and EQ2 models.


If you wish to run the Python code instead of the executable file, download both EasyQuadrat.rar and main.py files and follow these steps:

1. Make a new directory (e.g., "EasyQuadratProject") and place the main.py file there.
2. Unpack the EasyQuadrat.rar.
3. Navigate to the "..\EasyQuadrat\ _internal path."
4. Copy the following files to the newly made directory ("EasyQuadratProject"):

   a. EQ.ico (Application Icon)   
   b. EQS.png (Initializing Image)  
   c. Quadrat.png (GUI Image)  
   d. easyquadrat.pt (Model File)  
   e. easyquadrat2.pt (Another Model File)  

5. Open and run the main.py code in Python.
   

    
-------

Contact: Abbas Haghshenas  
Email: haqueshenas@gmail.com  
