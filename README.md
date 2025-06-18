Surveillance systems are widely used for monitoring and securing environments such as
public spaces, institutions, and private premises. Traditional surveillance requires
continuous human supervision, which is often inefficient, error-prone, and not scalable
for large volumes of video data. To overcome these limitations, this project proposes a
smart surveillance system capable of detecting abnormal human activities automatically.
The proposed system uses deep learning techniques to classify frames from surveillance
footage into two categories: normal and anomalous. It employs a Convolutional Neural
Network (CNN) model trained on the DCSASS dataset, which contains labeled instances
of various human activities. Once trained, the model is integrated into a user-friendly
interface built using Streamlit, allowing real-time image and video upload, processing,
and result display.
Upon detecting an anomaly, the system automatically logs the event and sends an alert
email to the registered user with visual evidence. It also includes secure user
authentication, an activity log database, and an admin panel for managing users and
system records. Designed for local deployment, this system serves as a practical, efficient,
and scalable solution to enhance existing surveillance mechanisms.

***************************************************************************

in smart_survilance.py add email and app password 

put admin.py in pages folder(create one)

dataset link: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
