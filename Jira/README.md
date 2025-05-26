# JIRA Integration API

A FastAPI-based RESTful API for interacting with JIRA to fetch and manage boards, epics, stories, tasks, descriptions, comments, attachments, and project users. The API provides endpoints to retrieve hierarchical data (boards → epics → stories → tasks/subtasks), manage issues, and save data to a JSON file.

---

##  Overview

This project is a Python-based FastAPI application that integrates with the JIRA REST API to fetch and manage project data. It allows users to:

* Retrieve JIRA boards, epics, stories, tasks, and subtasks in a hierarchical structure.
* Fetch and update issue descriptions, comments, and attachments.
* Create, update, and delete issues.
* Retrieve users associated with a JIRA project.
* Save the board-epic-story-task hierarchy to a JSON file.

The API uses the Atlassian JIRA REST API (version 3) and supports robust error handling, logging, and Atlassian Document Format (ADF) parsing for descriptions and comments.

---

##  Features

* **Hierarchical Data Retrieval:** Boards → Epics → Stories → Tasks/Subtasks with metadata.
* **Issue Management:** Create, read, update, delete issues.
* **Metadata Support:** Fetch/update descriptions, comments, attachments.
* **Project Users:** Retrieve users per JIRA project.
* **Data Export:** Save the full board hierarchy as `jira_hierarchy.json`.
* **Robust Error Handling:** Supports 401, 403, 404 with detailed logs.
* **Environment Configuration:** Secure setup via `.env` file.

---

##  Requirements

### Software

* Python 3.8+
* pip (Python package manager)
* Access to a JIRA Cloud instance with API token

### Python Dependencies

Create `requirements.txt` with:

```txt
fastapi
uvicorn
python-dotenv
requests
pydantic
python-multipart
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  JIRA API Access

* **JIRA\_EMAIL:** Your Atlassian email
* **JIRA\_API\_TOKEN:** Create from [https://id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
* **JIRA\_DOMAIN:** e.g., `yourcompany.atlassian.net`

Ensure permissions to:

* View/manage issues (epics, stories, tasks, subtasks)
* Access boards/projects
* Read/write descriptions, comments, attachments

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/jira-fastapi-api.git
cd jira-fastapi-api
```

### 2. Create `.env`

Create a `.env` file in the root:

```ini
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
JIRA_DOMAIN=your-domain.atlassian.net
```

### 3. Run the App

```bash
uvicorn main:app --reload
```

Swagger UI available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔗 API Endpoints

| Method | Endpoint                                | Description                           |
| ------ | --------------------------------------- | ------------------------------------- |
| GET    | `/boards`                               | Get all JIRA boards                   |
| GET    | `/boards/{board_id}/epics`              | Epics of a board with metadata        |
| GET    | `/epics/{epic_key}/stories`             | Stories and metadata of an epic       |
| GET    | `/stories/{story_key}/tasks`            | Tasks/subtasks for a story            |
| GET    | `/teams/project?project_key={key}`      | Users in a JIRA project               |
| GET    | `/hierarchy`                            | Board → Epic → Story → Task hierarchy |
| GET    | `/hierarchy/save`                       | Save hierarchy to JSON                |
| GET    | `/issues/{issue_key}/description`       | Get issue description                 |
| PUT    | `/issues/{issue_key}/description`       | Update issue description              |
| GET    | `/issues/{issue_key}/comments`          | List comments on an issue             |
| POST   | `/issues/{issue_key}/comments`          | Add comment to an issue               |
| GET    | `/issues/{issue_key}/attachments`       | List attachments                      |
| POST   | `/issues/{issue_key}/attachments`       | Upload an attachment                  |
| POST   | `/issues`                               | Create a new issue                    |
| GET    | `/issues?project_key={key}`             | List issues in a project              |
| PUT    | `/issues/{issue_id}`                    | Update issue summary                  |
| DELETE | `/issues/{issue_id}`                    | Delete an issue                       |
| GET    | `/issues/{issue_key}`                   | Fetch issue details                   |
| POST   | `/jira/api/{issue_key}/reassign_ticket` | Reassign issue to user                |
| DELETE | `/jira/api/{task_id}/closing_ticket`    | Mark issue as "Done"                  |

---

##  Output Example

```json
[
  {
    "id": 1,
    "name": "SCRUM board",
    "epics": [
      {
        "id": "10000",
        "name": "Build the Core Chatbot System",
        "description": "Integrate document search with OpenAI.",
        "comments": [...],
        "attachments": [...],
        "stories": [
          {
            "id": "10049",
            "key": "SCRUM-15",
            "summary": "Develop Chat API",
            "description": "Support auth, sessions, AI integration.",
            "tasks": [
              {
                "key": "SCRUM-16",
                "summary": "Design Chat REST API",
                "subtasks": [
                  {
                    "key": "SCRUM-17",
                    "summary": "Define API endpoints"
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
]
```

---

##  Code Structure & Highlights

* **Environment Validation:** `.env` loaded via `python-dotenv`. All credentials are validated on startup.
* **API Integration:** `requests` + `HTTPBasicAuth` to access JIRA endpoints.
* **Models:** Pydantic-based data models for Board, Epic, Story, Task, Subtask.
* **ADF Parsing:** Handles JIRA Atlassian Document Format (ADF) to read/write descriptions and comments.
* **Error Handling:** Centralized with status-aware responses (401, 403, 404, 500).
* **Reusability:** Utility functions like `extract_description`, `extract_comments`, `get_epic_link_field_id`, and `safe_request()` support maintainable code.

---

##  License

This project is licensed under the MIT License. See the LICENSE file for more information.


