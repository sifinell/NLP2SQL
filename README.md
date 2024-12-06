# NLP2SQL: Chat with Your SQL Database 🔍

Welcome to **NLP2SQL**, a Streamlit-based application that allows users to interact with a SQL database using natural language. Powered by Azure OpenAI, LangChain, and Vector Search, this tool simplifies database exploration with an intuitive chat interface.

## Features

- **Natural Language to SQL**: Convert plain language queries into SQL commands.
- **Embedded Knowledge Base**: Retrieve proper nouns like artist names or album titles using FAISS for more precise filtering.
- **Interactive Chat Interface**: Engage with the database through a user-friendly Streamlit chat interface.

## Files in the Repository

- **`.env`**: Contains environment variables for the Azure OpenAI configuration.
- **`requirements.txt`**: Lists required Python libraries.
- **`Chinook.db`**: A sample SQLite database.
- **`app.py`**: The main application file that connects LangChain, Azure OpenAI, and the Chinook database.

## Installation and Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/sifinell/NLP2SQL.git
    cd NLP2SQL
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the environment variables**:
   - Create a `.env` file in the project root with the following content:

    ```plaintext
    AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
    AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
    AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
    AZURE_OPENAI_API_VERSION="2024-05-01-preview"
    ```

   - Replace `your-azure-openai-endpoint` and `your-azure-openai-api-key` with your Azure OpenAI deployment details.

4. **Run the application**:

    ```bash
    streamlit run app.py
    ```

## Sample Questions

Here are some examples of queries you can ask the application:

1. **List all artists**:

    ```plaintext
    Input: List all artists.
    Query: SELECT * FROM Artist;
    ```

2. **Find all albums for the artist 'AC/DC'**:

    ```plaintext
    Input: Find all albums for the artist 'AC/DC'.
    Query: SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');
    ```

3. **List all tracks in the 'Rock' genre**:

    ```plaintext
    Input: List all tracks in the 'Rock' genre.
    Query: SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');
    ```

4. **Find the total duration of all tracks**:

    ```plaintext
    Input: Find the total duration of all tracks.
    Query: SELECT SUM(Milliseconds) FROM Track;
    ```

5. **List all customers from Canada**:

    ```plaintext
    Input: List all customers from Canada.
    Query: SELECT * FROM Customer WHERE Country = 'Canada';
    ```

6. **How many employees are there?**:

    ```plaintext
    Input: How many employees are there?
    Query: SELECT COUNT(*) FROM "Employee";
    ```

## References

For more details about the LangChain SQL capabilities used in this project, visit the [LangChain SQL Quickstart documentation](https://python.langchain.com/docs/tutorials/sql_qa/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Feel free to submit issues or pull requests for new features and improvements!
