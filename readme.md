## Digital-Agriculture
Recently we have observed the emerging concept of smart farming that makes agriculture more efficient and effective with the help of high-precision algorithms. The mechanism that drives it is Machine Learning — the scientific field that gives machines the ability to learn without being strictly programmed. 
It has emerged together with big data technologies and high-performance computing to create new opportunities to unravel, quantify, and understand data intensive processes in agricultural operational environments.

Machine learning is everywhere throughout the whole growing and harvesting cycle. 
It begins with a seed being planted in the soil — from the soil preparation, seeds breeding and water feed measurement — and it ends when neural networks pick up the harvest determining the ripeness with the help of computer vision.

### Data Description
Each training and test example is assigned to one of the following labels:

| Variable                | Description                                                                            |
|-------------------------|----------------------------------------------------------------------------------------|
| ID                      | UniqueID                                                                               |
| Estimated_Insects_Count | Estimated insects count per square                                                     |
| Crop_Type               | Category of Crop (0,1)                                                                 |
| Soil_Type               | Dress                                                                                  |
| Pesticide_Use_Category  | Type of pesticides uses (1-Never,2-Previously Used,3-Currently Using)                  |
| Number_Doses_Week       | Number of doses per week                                                               |
| Number_Weeks_Used       | Number of weeks used                                                                   |
| Number_Weeks_Quit       | Number of weeks quit                                                                   |
| Season                  | Season Category (1,2,3)                                                                |
| Crop_Damage             | Crop Damage Category (0=alive,1=Damage due to other causes,2=Damage due to Pesticides) |

## Project Structure

```
.
├── custom_models   
│   └── net_model.py 
├── data   
│   └── agro_net.h5
│   └── agro_net.png
│   └── Digital-Argiculture.ipynb
│   └── Digital_Argiculture.ipynb
│   └── final-submission.csv
│   └── final_prediction.csv
│   └── sample_submission.csv
│   └── test.csv
│   └── train.csv
├── main.py
├── README.md
├── requirements.txt
```

## Usage

```
python3 main.py 
```

## License
This project is licensed under the terms of the [MIT license](https://choosealicense.com/licenses/mit/).
