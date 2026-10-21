FaceLivenessPAD

Face Liveness Detection and Presentation Attack Detection for Biometric Authentication

Project Type: Biometric Authentication Security Prototype
Programming Language: Python
Primary Goal: Face Recognition, Face Liveness Detection, and Presentation Attack Detection (PAD)

---

1. Project Overview

FaceLivenessPAD is a Python-based biometric authentication prototype designed to verify a person's identity and assess whether a face presented to a webcam appears to be live or a possible presentation attack.

Traditional face recognition can match facial features without necessarily determining whether the input comes from a live person, a printed photograph, or a replay displayed on a screen. This project combines face matching with passive liveness analysis, Presentation Attack Detection (PAD), and active facial challenges to improve the authentication process.

The system captures facial images through a webcam, extracts facial features, compares the captured face against registered facial data, evaluates visual and movement-based indicators, and displays an authentication status.

The project also includes resident registration, operator and supervisor authentication, administrative setup, and audit logging.

2. Problem Statement

Face recognition systems may be vulnerable to presentation attacks in which someone presents a photograph, a face displayed on a mobile phone, or a replayed video to the camera.

The objective of FaceLivenessPAD is to combine identity verification and anti-spoofing checks in one prototype to make face-based authentication more resistant to these attacks.

3. Project Objectives

- Verify a captured face against a registered resident's facial data.
- Generate and compare numerical facial embeddings.
- Analyze facial movements and visual characteristics.
- Assess potential presentation attacks using a PAD model.
- Support active challenges such as blinking, smiling, and turning the head.
- Provide operator and supervisor authentication modules.
- Support administrative setup and audit logging.
- Provide clear authentication statuses for demonstration and testing.

4. Technology Stack

Technology| Purpose
Python| Main application programming language
OpenCV| Webcam access, image processing, and display
MediaPipe Face Mesh| Facial landmark detection and movement analysis
InsightFace| Face analysis and facial embedding extraction
"buffalo_l"| InsightFace model package used for face analysis
NumPy| Numerical operations and embedding comparison
ONNX| Model interchange format used by the PAD model
ONNX Runtime| Loads and executes the PAD model
MiniFASNetV2| Pretrained model used for presentation attack detection
Farnebäck optical flow| Estimates motion between video frames
Laplacian variance| Estimates image sharpness and texture detail

Note: MiniFASNetV2 is the pretrained PAD model currently used in the project. Training a custom PAD model is a planned improvement, not a feature claimed to be completed.

5. System Architecture

The application coordinates the different modules through "main.py".

                    WEBCAM INPUT
                         |
                         v
                  Face Detection
                         |
                         v
              Face Embedding Extraction
                  using InsightFace
                         |
                         v
              Identity Matching Check
                         |
                         v
              +----------------------+
              |                      |
              v                      v
      Passive Liveness         PAD Inference
      Motion and Texture       MiniFASNetV2
      Facial Indicators        ONNX Runtime
              |                      |
              +----------+-----------+
                         |
                         v
                Active Face Challenge
              Blink / Smile / Head Turn
                         |
                         v
                Authentication Logic
                         |
                         v
                 Final Status Display

The system also contains supporting modules for resident registration, operator authentication, supervisor authentication, admin setup, and audit logging.

6. How the Project Works

Step 1: Webcam Capture

OpenCV accesses the webcam and captures video frames. The frames are processed continuously so that the system can analyze the face and its movements.

Step 2: Face Detection and Landmark Analysis

The application detects the face in the captured frame. MediaPipe Face Mesh identifies facial landmarks around regions such as the eyes, mouth, and face outline.

These landmarks help the application analyze facial geometry and movements for supported checks and active challenges.

Step 3: Face Embedding Generation

InsightFace processes the captured face and generates a numerical representation called a face embedding.

An embedding represents facial features as a vector of numbers. It is not simply the original photograph.

Step 4: Identity Matching

The application compares the captured embedding with the embedding stored for the registered resident.

The "face_match.py" module normalizes the embeddings and calculates cosine similarity.

The current configured identity threshold is "0.70". A score at or above this threshold is treated as a match by the current implementation; a score below it is not treated as a match.

This is a configurable project threshold, not a universal guarantee of identity recognition accuracy.

Step 5: Passive Liveness Analysis

The application analyzes several visual and movement-based indicators, including:

- Changes between video frames.
- Motion estimated using Farnebäck optical flow.
- Facial landmark geometry and movement.
- Image sharpness estimated using Laplacian variance.
- Texture-related indicators.

These indicators contribute to the passive liveness assessment. They cannot independently guarantee that an input is from a live person.

Step 6: Presentation Attack Detection (PAD)

The application uses the pretrained MiniFASNetV2 model in ONNX format.

The face crop is resized to the model's expected input size of 80 × 80 pixels. ONNX Runtime executes the model, and the application interprets its output to assess whether the input is classified as real or spoof.

PAD is intended to help identify presentation attacks such as printed photographs and faces displayed on screens. Its reliability depends on the model, preprocessing, test conditions, and attack types.

Step 7: Active Liveness Challenges

The application supports active facial challenges, including:

- Blink.
- Smile.
- Turn head left.
- Turn head right.

The application analyzes facial landmarks and movements to determine whether the requested action has been performed.

Active challenges provide additional evidence but do not guarantee resistance to every possible attack.

Step 8: Final Authentication Status

The main application combines identity matching, PAD, passive liveness, and challenge results.

The current status logic includes:

Status| Meaning
"LIVE"| Identity matched, PAD classified the input as real, passive checks are ready, and the active challenge passed
"SPOOF"| PAD classified the input as a spoof
"IDENTITY FAIL"| The captured identity did not match the registered identity
"CHECKING"| The required checks are not yet complete

These are application statuses, not a certification of biometric security.

7. Main Project Modules

File or module| Responsibility
"main.py"| Main application, camera processing, checks, and final status logic
"face_match.py"| Face embedding generation and identity comparison
"operator_login.py"| Operator authentication functionality
"supervisor_login.py"| Supervisor authentication functionality
"admin_setup.py"| Administrative setup functionality
"audit_logger.py"| Recording application events in the audit log
"models/MiniFASNetV2.onnx"| Pretrained PAD model file

The actual files and names in the repository should be checked before final submission.

8. Resident Registration

The registration functionality stores the facial data required for later identity matching.

In the current implementation, an embedding can be saved when it is available. Registration is not currently gated on every passive-liveness, PAD, and active-challenge check passing.

For a real deployment, registration should require appropriate identity checks, liveness/PAD checks, consent, and secure handling of biometric information.

9. Authentication and Administration

The project includes supporting modules for:

- Operator login.
- Supervisor login.
- Administrative setup.
- Resident data management.
- Audit-event recording.

The application is a prototype. Before deployment, administrative access should use secure credential storage, appropriate access controls, and protected audit logs.

Demo credentials: Do not place passwords or secret credentials in this public README. Supply any authorised demonstration credentials through a private channel.

10. Project Directory Structure

The following illustrates the intended project organisation. Confirm the actual directory and file names against the source code before relying on this example.

FaceLivenessPAD/
|
├── main.py
├── face_match.py
├── operator_login.py
├── supervisor_login.py
├── admin_setup.py
|
├── audit_logger.py
|
├── admin_setup.py
├── models/
|   └── MiniFASNetV2.onnx
|
├── data/
|   ├── users/
|   ├── operator/
|   └── supervisor/
|
├── logs/
|
├── requirements.txt
└── README.md

Do not commit actual resident photographs, biometric embeddings, private credentials, or sensitive audit records.

11. Requirements

The development environment used for the project includes:

- Python 3.12.
- A working webcam.
- OpenCV.
- MediaPipe.
- InsightFace.
- NumPy.
- ONNX Runtime.
- The required PAD model file.
- Other dependencies required by the project's modules.

Exact dependency versions should be documented in "requirements.txt" based on the environment in which the application has been tested.

12. Installation and Execution

1. Install a compatible Python version.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install the dependencies from "requirements.txt".
5. Ensure that the PAD model is available at the path expected by the application.
6. Connect a webcam or enable the built-in camera.
7. Run the application:

python main.py

The actual installation commands, model setup, and operating-system requirements must be verified against the final project files before submission.

13. Test Plan

The following cases should be tested and the actual outcomes recorded.

Test case| Expected behaviour
Registered resident| Identity matching succeeds when the score meets the configured threshold
Unregistered person| Identity matching fails if no stored embedding meets the threshold
Printed photograph| PAD should attempt to detect the presentation attack
Face displayed on a phone| PAD should attempt to detect the presentation attack
Blink challenge| The application checks for the required blink
Smile challenge| The application checks for the required smile
Head-turn challenge| The application checks the requested head movement
Poor lighting| Observe and document the effect on detection and matching
No face in frame| The application should not report a successful identity match without a detected face

Expected behaviour is not the same as measured performance. Actual outcomes must be recorded after testing.

14. Evaluation Metrics

For a more complete technical evaluation, report:

- Accuracy: Overall proportion of correct predictions.
- APCER: Attack Presentation Classification Error Rate.
- BPCER: Bona Fide Presentation Classification Error Rate.
- False-match rate: Frequency of incorrect identity matches.
- Processing latency: Time required to process and return a result.

No numerical performance claims are made here because verified evaluation results have not been provided.

15. Security and Privacy

Facial photographs and embeddings are sensitive biometric data. The prototype should be tested with consented or synthetic data wherever possible.

Security considerations include:

- Protecting stored embeddings and resident records.
- Avoiding hardcoded passwords and public credentials.
- Restricting access to operator and supervisor functions.
- Protecting audit logs from unauthorised access.
- Reviewing third-party model licences.
- Avoiding claims of production-grade security without formal testing.

16. Limitations

- Results can vary with lighting, camera quality, pose, and image quality.
- An embedding similarity threshold does not guarantee correct identity recognition.
- Facial landmarks alone cannot establish that a face is live.
- A pretrained PAD model may not detect every presentation attack.
- The configured threshold and model output require validation on representative test data.
- The current registration workflow does not require all liveness and PAD checks to pass before saving an embedding.

17. Future Scope

- Train and evaluate a custom PAD model.
- Test against a wider range of presentation attacks.
- Improve secure biometric storage and access control.
- Record quantitative performance metrics.
- Improve usability and error handling.
- Validate the application across different cameras and lighting conditions.

18. Conclusion

FaceLivenessPAD demonstrates an integrated approach to face-based authentication by combining facial embedding comparison, passive liveness indicators, a pretrained PAD model, and active facial challenges.

The project also provides supporting registration and administrative modules. Further dataset-based evaluation, secure data handling, and custom PAD model development are planned improvements before any production use.

19. References and Resources

- InsightFace: https://github.com/deepinsight/insightface
- MediaPipe: https://github.com/google-ai-edge/mediapipe
- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/

Project status: Prototype developed; custom PAD model training and comprehensive performance evaluation remain future work.