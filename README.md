# Lab Abnormalities as Predictors for Chronic Diseases

A Streamlit-based Clinical Insight Demonstration

## Try the Application Online (No Installation Required)

Use the app instantly through Streamlit Cloud:

## Live Application:

https://lababnormalitiesaspredictorsforchronicdiseases-hy5wt3w2smugaww.streamlit.app

This hosted version allows you to upload patient JSON files and view model-generated risk reports directly in your browser.

## Using the Web App

On the Upload page, you now have two ways to try the application:

## Method 1 — Use Built-in Sample Patients (Recommended for quick demos)

For quick testing, you can use the bundled sample patient files stored in the sample_data/ folder
(named patient_1.json, patient_2.json, etc.)

On the Upload page:

Select a patient from the dropdown (e.g., patient_1).

Click “Load sample into app” to immediately analyze that patient.

(Optional) Click “Download sample JSON” to save the file for Method 2.

Why this is helpful:

No need to download synthea data generator from GitHub for sample data

Instantly explore how the app works using realistic synthetic patients

Perfect for demonstrations, teaching, and testing

## Method 2 — Upload Your Own FHIR JSON

If you want to generate your own synthetic FHIR patient data using Synthea:

Step 1: Download Synthea

Download the Synthea JAR file from GitHub:

https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

Step 2: Navigate to the download location for synthea-with-dependencies.jar

Example:

cd ~/Downloads

Step 3: Ensure Java is installed

java -version

Step 4: Generate patients

Example command:

java -jar ~/Downloads/synthea-with-dependencies.jar \
  -p 100 \
  --modules \
  diabetes_mellitus_type_2,chronic_kidney_disease,cardiovascular_risk,anemia,liver_disease \
  --exporter.baseDirectory="$HOME/synthea_output" \
  California

Where:

-p = number of patients to generate

--modules = list of chronic conditions to simulate

--exporter.baseDirectory = output folder

Last argument (California) = state to simulate population

Step 5: Upload your generated JSON

Go to the Upload page

Use the “Upload your lab report (FHIR JSON)” uploader

After a successful upload → view your results in Report Analysis and Recommendations

## Project Purpose

This project demonstrates how lab abnormalities can be used as predictors for several chronic diseases:

Prediabetes / Diabetes

Chronic Kidney Disease (CKD)

Cardiovascular Disease (CVD)

Anemia

The application provides:

An interactive Streamlit interface

Patient-friendly risk visualizations

Lab-by-lab interpretations

Insights based on clustering + ML models

Support for both uploaded real/synthetic data and built-in sample patients

Designed for education, research, and demonstration, it translates complex lab data into understandable patterns.

## Disclaimer

This tool is for educational and research use only.
It is not intended for:

Clinical diagnosis

Treatment decisions

Professional medical advice

Always consult a licensed healthcare professional for real medical concerns.