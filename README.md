
# End-to-End Electric Vehicle (EV) Insights Dashboard

Welcome to the EV Insights Dashboard! This project is a complete end-to-end MLOPS mini-system designed to analyze the electric vehicle market. It provides functionalities for EV price prediction, in-depth exploratory data analysis (EDA), and an intelligent chatbot for querying the dataset.

This project is built with a clean, object-oriented architecture and is intended to be a practical learning resource for students and enthusiasts interested in MLOPS and data science.

## 🚀 Features

-   **EV Price Prediction**: Predict the Base MSRP (Manufacturer's Suggested Retail Price) of an electric vehicle using machine learning.
    -   Models: Simple Linear Regression and Random Forest Regressor.
    -   Evaluation: View key performance metrics like MSE, MAE, R-squared, and RMSE.
-   **Exploratory Data Analysis (EDA)**: An interactive dashboard with various plots to visualize and understand the EV dataset.
    -   Data distributions, correlations, and categorical analysis.
    -   Visualizations are created using `matplotlib`.
-   **EV Chatbot**: A conversational AI, powered by LangChain and OpenAI, that can answer your questions about the EV dataset.
-   **Clean Architecture**: The project follows a modular, service-oriented architecture with clear separation of concerns, making it easy to understand and extend.

## 📂 Project Structure

The project uses a `src` layout, which is a standard practice in modern Python projects. This helps in organizing the code in a clean and modular way.

```
.
├── README.md
├── requirements.txt
└── src
    ├── chatbot
    │   └── chatbot_service.py
    ├── datasets
    │   └── ev_raw_data.csv
    ├── ml_models
    │   └── regression.py
    ├── services
    │   └── data_service.py
    ├── streamlit_app
    │   ├── main_app.py
    │   └── pages
    │       ├── 1_prediction.py
    │       ├── 2_eda.py
    │       └── 3_chatbot.py
    └── utils
        └── eda.py
```

-   **`src/`**: The main source code directory.
    -   **`chatbot/`**: Contains the `ChatbotService` for the LangChain-powered chatbot.
    -   **`datasets/`**: Stores the raw data files.
    -   **`ml_models/`**: Holds the regression model classes.
    -   **`services/`**: Contains the `DataService` for data loading and processing.
    -   **`streamlit_app/`**: The main Streamlit application, with each page in the `pages/` subdirectory.
    -   **`utils/`**: Utility functions, such as the `EDAUtils` for generating plots.
-   **`requirements.txt`**: A list of all Python dependencies.
-   **`README.md`**: This file!

## 🛠️ Setup and Installation

This project uses `uv` for package management, which is a fast and modern alternative to `pip` and `venv`.

### Prerequisites

-   Python 3.8 or higher.
-   `uv` installed. If you don't have it, you can install it with:
    ```bash
    pip install uv
    ```

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    `uv` makes this a one-step process.
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
    This will create a `.venv` directory with all the necessary packages installed.

3.  **Activate the virtual environment:**
    -   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

### 🤖 Configuring the Chatbot (Important!)

The chatbot requires an OpenAI API key to function. This is managed using a `.env` file in the root of the project.

1.  **Locate the `.env` file**: In the main project directory, you will find a file named `.env`.

2.  **Add Your API Key**: Open the `.env` file and replace `"YOUR_API_KEY_HERE"` with your actual OpenAI API key.
    ```
    OPENAI_API_KEY="sk-..."
    ```

3.  **Get an API Key**: If you don't have one, you can get an API key from the [OpenAI Platform](https://platform.openai.com/account/api-keys).

The application will automatically load the key from this file. Your key is kept secure on your local machine and is not shared.

## ▶️ How to Run the Application

Once you have completed the setup and activated the virtual environment, you can run the Streamlit application with a single command:

```bash
streamlit run src/streamlit_app/main_app.py
```

This will start the web server and open the EV Insights Dashboard in your default web browser.

## 📝 Sample Test Run Example

Here’s how you can interact with the application:

1.  **Navigate to the Prediction Page**:
    -   Select "Random Forest Regressor" from the dropdown.
    -   Observe the performance metrics displayed.
    -   Choose a `Make` (e.g., "TESLA"), a `Model` (e.g., "MODEL Y"), and an `Electric Vehicle Type`.
    -   Enter an `Electric Range` (e.g., 300 miles).
    -   Click the **"Predict Price"** button. The predicted MSRP will be displayed.

2.  **Explore the EDA Page**:
    -   Scroll through the page to see different visualizations of the dataset.
    -   Analyze the distribution of EV prices, ranges, and the popularity of different makes.

3.  **Chat with the Bot**:
    -   Ensure you have added your OpenAI API key to the `.env` file as described in the configuration section.
    -   Go to the **Chatbot** page.
    -   Ask a question in the chat input, for example:
        -   "What is the average electric range for TESLA cars?"
        -   "Which car has the highest Base MSRP?"
        -   "Tell me about the NISSAN Leaf."
    -   The chatbot will provide an answer based on the dataset. You can expand the "See sources" section to see which data points were used to generate the answer.

---

We hope you find this project useful and educational. Happy coding!
