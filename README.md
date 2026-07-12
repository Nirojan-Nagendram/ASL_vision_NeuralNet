# ASL alphabet computer vision using Google's MediaPipe Landmarker

This project uses Google's pretrained landmarked hand recognising AI, to train a separate neural network trained on the ASL alphabet. This is then combined with a live webcam to display results in real time, using async threads & callback functions.

## MediaPipe, callbacks & live results
MediaPipe is a well documented & accurate model, the hand landmarks are very impressive and made this project feasible. The amount of data required to train these landmarkers is far beyond my means. The combination with OpenCV made this project much easier to diagnose issues visually. The livestream variation of MP is similar to frontend JSX React in that they both heavily rely on callback functions, which was an interesting challenge. The code could not wait to send and receive the model's prediction. Therefore it requires threading to allow for uninterrupted webcam with live results.

## How to run

### On a standard computer
1) Install dependencies:
```bash
pip install -r requirements.txt
```
2) Download `hand_landmarker.task` from the MediaPipe docs and place it in the project folder
3) Collect training data by running `data_collection.py` — press a letter key to start collecting that letter, repeat for all letters, or use the included training data
4) Train and start the Flask server by running `app.py`
5) Run `main.py` to open the webcam and see live predictions

### On a Raspberry Pi 5
1) Flash Raspberry Pi OS (Legacy, 64-bit) using Raspberry Pi Imager — enable SSH, set a username and password in the OS customisation settings before flashing
2) Connect your USB camera and DSI display to the Pi, then power on
3) SSH into the Pi from your laptop: `ssh username@hostname.local`
4) Clone the repository and install dependencies:
```bash
git clone https://github.com/Nirojan-Nagendram/ASL_VisionNeuralNet
cd ASL_VisionNeuralNet
pip install -r requirements.txt
```
5) Download `hand_landmarker.task` from the MediaPipe docs and place it in the project folder
6) Start the Flask server in one terminal: `python3 app.py`
7) Run the main script in a second terminal: `python3 main.py`

## Development diary (Version control with Git/GitHub)
01/07:
- First commit, first draft finished. I made the model with tensorflow & a flask API to connect to. I wrote a script to capture webcam screenshots corresponding to specific letters. I screenshotted every frame or every few to capture many different poses of the same letter. I recorded each letter, saving them to a separate file. I then wrote the main webcam script. I used the threading library to make a callback function which would send and wait for a response. This did require some extra checks for empty frames but allowed for constant live responses.

- This led to a very inaccurate model albeit free of technical bugs. The first and most obvious issue was Z & J,*(1. ASL alphabet)*, the letters involve movement in ASL. This does not work with my current framework. It takes a single image's landmarks and trains with that. It would need to be a couple consecutive frames. I did attempt to do the motion and screenshot throughout, labelling them Z & J. However, this meant many different gestures were labelled Z & J, which meant they would appear in a disproportonate number of results. I temporarily removed J & Z, this improved the accuracy.*(2. 1st commit - working)*

- Next issue was amount of data, originally I had around 100 sets of landmarks, for a model with 26 outputs this is just not enough data to learn. So i used prepare_dataset.py to increase the total to 5000 sets. I also repeated with slightly different gestures of the same letter, not just different frames to ensure more variety in the training data. As the landmarks would change with different people, due to signing styles and proportions of fingers, so I need prepare some data of other people signing. *(3. 1st commit - not working)*

09/07:
- I am preparing to run the project on a Raspberry pi. In addition, I reordered the requirements to make sure that mediapipe was downloaded last, and the correct version of torch for a Pi was also added.
- To optimise for the Raspberry Pi's limited CPU, I replaced the live webcam feed with a black canvas — the hand skeleton drawn from landmarks communicates the hand position just as effectively. I also reduced the camera resolution to suit the smaller screen and cut down on data per frame. A custom universal logger was added to simplify debugging across scripts. There is still significant lag, I need to reduce the load of MediaPipe to improve the lag. Currently it is unusable. *(4. 2nd / 3rd commit - Pi 5 integration - lagging)*

11/07:
- My next goal is to reduce CPU load on the Pi without relying on additional hardware. MediaPipe is the primary bottleneck, so I reduced how frequently frames are sent to it by processing every other frame — this introduces minimal visual lag while halving the workload. I also added a check to prevent multiple prediction threads running simultaneously, avoiding unnecessary CPU load. Finally I reduced minimum tracking confidence within MediaPipe, this will make it less accurate but cost less overhead.
12/07:
 - After testing, the CPU temperature reduced from 74°C to 62°C. This is much more managable with just heatsinks, rather than a fan. Additionally, the video has stopped lagging beyond the expected amount due to accessing MP every other frame. It is much more usable. I added a flip to the frame, as the Pi sits better on that side. Reducing the minimum tracking confidence has seem to increase accuracy specifically when the fingers are close together, it does not seem to lag much without it. So I am leaving it at the regular minimum confidence for now. 
 - I am working on casing for the camera, however the neural network's accuracy is the biggest issue. It is more accurate with letters where the fingers do not overlap such as b or u-w. However it struggles with letters such M and N. There are a good number of very similar letters, which have one finger (usually the thumb) in a different place. Additionally, the angle of the camera matters a lot, for example the letters U & V get confused as from the side they look very similar in 2D plane.*(5. U to V turning)*


## Screenshots:

1. ASL alphabet:

![ASL Alphabet](images/ASL_Alphabet.jpg)

2. 1st commit - working:

![1st commit - working](images/1st_commit_screenshot.png)

3. 1st commit - not working: 

![1st commit - not working](images/1st_commit_incorrect.png)

4. 2nd / 3rd commit - Pi 5 integration - lagging:

![Pi 5 integration - lagging](images/pi_integration_lagging.gif)

5. U to V turning

![U to V turning](images/U_to_V_turning.gif)

## Known limitations
 - Missing letters J & Z, currently not possible with current framework.
 - Inaccurate results
 - Only registers first hand in shot, not a problem for ASL alphabet but would be a problem with words or BSL. It would be a problem if multiple people signed at once as well. Google's landmarks include number of hands, so this is solvable.

## Future plans
- improve accuracy, better & more training data with more variety
- Once model is accurate enough, save model trained so it doesn't need to be retrained.
- Deploythe model to raspberry pi for portable camera