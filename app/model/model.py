import pickle
import re
import hashlib
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


# Handle categorical variables
label_enc_gender = LabelEncoder()
label_enc_gender.fit(['Male', 'Female'])  # Pre-fit with known categories

label_enc_geography = LabelEncoder()
label_enc_geography.fit(['France', 'Germany', 'Spain'])  # Pre-fit with known categories

# Define a function to hash surnames to a consistent integer value
def hash_surname(surname):
    # Use md5 to get a consistent hash across platforms
    hash_obj = hashlib.md5(str(surname).encode())
    # Convert first 4 bytes of hash to integer and take modulo 1000 
    # to limit to 0-999 range
    return int(hash_obj.hexdigest()[:8], 16) % 1000


def predict_pipeline(customer_data: dict):
    try:
        # Log input data for debugging
        logging.info(f"Received customer data: {customer_data}")
        
        # Validate required fields
        required_fields = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                          'Balance', 'NumOfProducts', 'HasCrCard', 
                          'IsActiveMember', 'EstimatedSalary', 'Surname']
        
        for field in required_fields:
            if field not in customer_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert input dictionary to list of features in correct order
        # Use label encoding for gender and geography
        logging.info(f"Processing gender: {customer_data['Gender']}")
        
        # Ensure gender and geography are strings
        gender = str(customer_data["Gender"])
        geography = str(customer_data["Geography"])
        
        # Transform single values using pre-fitted encoders
        try:
            gender_encoded = label_enc_gender.transform([gender])[0]
            logging.info(f"Gender encoded: {gender} -> {gender_encoded}")
        except Exception as e:
            logging.error(f"Error encoding gender: {e}")
            raise ValueError(f"Invalid gender value: {gender}. Must be one of: Male, Female")
        
        try:
            geography_encoded = label_enc_geography.transform([geography])[0]
            logging.info(f"Geography encoded: {geography} -> {geography_encoded}")
        except Exception as e:
            logging.error(f"Error encoding geography: {e}")
            raise ValueError(f"Invalid geography value: {geography}. Must be one of: France, Germany, Spain")
        
        # Process surname directly without using apply
        surname_encoded = hash_surname(customer_data["Surname"])
        logging.info(f"Surname encoded: {customer_data['Surname']} -> {surname_encoded}")
        
        # Validate numeric fields
        try:
            credit_score = int(customer_data['CreditScore'])
            if not 300 <= credit_score <= 850:
                logging.warning(f"Credit score outside normal range: {credit_score}")
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid credit score value: {customer_data['CreditScore']}")
            raise ValueError(f"CreditScore must be a valid integer between 300 and 850") from e
        
        try:
            age = int(customer_data['Age'])
            if not 18 <= age <= 100:
                logging.warning(f"Age outside normal range: {age}")
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid age value: {customer_data['Age']}")
            raise ValueError(f"Age must be a valid integer between 18 and 100") from e
            
        # Validate additional numeric fields
        numeric_fields = {
            'Tenure': {'min': 0, 'max': 10},
            'Balance': {'min': 0},
            'NumOfProducts': {'min': 1, 'max': 4},
            'HasCrCard': {'values': [0, 1]},
            'IsActiveMember': {'values': [0, 1]},
            'EstimatedSalary': {'min': 0}
        }
        
        for field, constraints in numeric_fields.items():
            try:
                value = float(customer_data[field]) if field in ['Balance', 'EstimatedSalary'] else int(customer_data[field])
                
                if 'values' in constraints and value not in constraints['values']:
                    logging.error(f"Invalid {field} value: {value}. Must be one of {constraints['values']}")
                    raise ValueError(f"{field} must be one of {constraints['values']}")
                    
                if 'min' in constraints and value < constraints['min']:
                    logging.warning(f"{field} below minimum value: {value} < {constraints['min']}")
                    
                if 'max' in constraints and value > constraints['max']:
                    logging.warning(f"{field} above maximum value: {value} > {constraints['max']}")
                    
                customer_data[field] = value
                
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid {field} value: {customer_data[field]}")
                raise ValueError(f"{field} must be a valid number") from e
                
        # Create feature vector in the correct order
        features = [
            customer_data['CreditScore'],
            geography_encoded,
            gender_encoded,
            customer_data['Age'],
            customer_data['Tenure'],
            customer_data['Balance'],
            customer_data['NumOfProducts'],
            customer_data['HasCrCard'],
            customer_data['IsActiveMember'],
            customer_data['EstimatedSalary'],
            surname_encoded
        ]
        
        logging.info(f"Feature vector types: {[type(f) for f in features]}")

        logging.info(f"Final feature vector: {features}")
        
        # Get probability predictions
        try:
            prob = model.predict_proba([features])[0]
            churn_probability = prob[1]  # Probability of class 1 (churn)
            
            # Make final prediction (True if probability > 0.5)
            will_churn = churn_probability > 0.5
            
            logging.info(f"Prediction result: probability={churn_probability:.4f}, will_churn={will_churn}")
            
            return {
                "churn_probability": float(churn_probability),
                "will_churn": bool(will_churn)
            }
        except Exception as e:
            logging.error(f"Error during model prediction: {str(e)}")
            raise ValueError(f"Error during prediction: {str(e)}")
        
    except Exception as e:
        logging.error(f"Error in predict_pipeline: {str(e)}", exc_info=True)
        raise ValueError(f"Prediction failed: {str(e)}")
