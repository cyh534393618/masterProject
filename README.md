# MasterProject

## Description
The source code of this project is mainly divided into two parts: the front end and the back end. The front end is primarily developed using HBuilder X, while the back end is developed using Node.js and Nest.js. The Python scripts in the 'Python Script' folder are already included in the BackEnd/Py folder for easy reference to the relevant scripts.

## Front End setup
Step 1: Download HBuilder X and open the source code project in the FrontEnd folder.

Step 2: Install an Android emulator on your local machine, such as LDPlayer.

Step 3: In HBuilder X, click the run button and select 4. Run to Android App Base.

Step 4: Please ensure that the Android emulator is correctly opened on your local machine. Select the emulator from the list and click the run button.

<img src="READMEimg/Image_20241021171801.png" alt="Step 3-4" width="80%">

Step 5: If everything is working correctly, the program will run properly in the emulator, as shown in the image below.

<img src="READMEimg/Image_20241021173058.png" alt="Step 5" height="60%">

## Back End setup
Step 1: Please ensure that Node.js is correctly installed and configured on your local machine.

Step 2: Open the source code project in the BackEnd folder using Visual Studio Code.

Step 3: In the VS Code terminal, run the npm init command to download the necessary packages. Once the download is complete, a node_modules folder will be created in the root directory.

<img src="READMEimg/Image_20241021174808.png" alt="Step 2-3" width="80%">

Step 4: Run the npm start command in the terminal to start the back end service.

<img src="READMEimg/Image_20241021174946.png" alt="Step 4" width="80%">

Step 5: Please use the ipconfig command to determine your local machine's IP address.

<img src="READMEimg/Image_20241021175237.png" alt="Step 5" width="80%">

Step 6: Open the front end project in HBuilder X. Open the pages/index/index.vue file and change the IP address on line 88 to your local machine's IP address, with the port number set to 3000.

<img src="READMEimg/Image_20241021175502.png" alt="Step 6" width="80%">
