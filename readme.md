# Visual Search Demo
### Rank image similarity of cars by using Google's Inception Model to generate image features.
Created By: Rohan Doshi

Demo: http://VisualSearch.pythonanywhere.com/

#### Description
This is a visual search demo. Click on a car, and the other 19 cars will re-order by similarity. This is done by sorting the pair-wise euclidian distance between the feature vectors for the images. We generate these feature vectors using a pre-trained TensorFlow implementation of Google's Inception-v3 model. Although the Inception-v3 model is originally trained for a general 1000-label classification task, I postulate the second-to-last layer containing a 2048 dimension feature vector feeding into the softmax classifier can be used as a generalizable representation of an image (a.k.a. transfer learning). To learn more about the underlying convolutional neural network architecture for Inception-v3, check out this paper by Szegedy et al at https://arxiv.org/abs/1512.00567.

#### How to Run
1. Clone the repo
2. Change your director.y to the folder holding server.py
2. Launch the flask server using the bash command: python server.py
3. Open the app by navigating to http://127.0.0.1:5000/ on a web-browser (e.g. Google Chrome)

#### Design Decisions
I implement the Python backend using the Flask-framework in order to easily interface with the Python API for TensorFlow, Google's machine learning library. For the production deployment of the app, I've cached all the image feature vectors in a JSON object to
prevent the need to generate the TensorFlow graph and input all the images each time the home page loads (~10 second process). I've decided to implement the pairwise euclidian distance calculations on the client-side as opposed to the server-side in order to eliminate the transmission time of HTTP request between the client and server; there are only twenty samples, so the client can handle this level of computation. In terms of my choice of vector similarity metrics for feature vectors, I have experimentally found that euclidian distance works better than cosine similarity, which may be attributed to the high dimensionality of the vectors.

#### Credit
1. TensorFlow's Image Classification Tutorial (https://www.tensorflow.org/tutorials/image_recognition/)