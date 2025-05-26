import os
import json
from fastapi import File, UploadFile, FastAPI, HTTPException, Form,APIRouter, Query,Body
from requests.auth import HTTPBasicAuth
from typing import List, Optional, Any
from dotenv import load_dotenv
import requests
import base64
from pydantic import BaseModel, HttpUrl, Field
from pydantic import validator
# Load environment variables from .env
load_dotenv()

# Validate environment variables
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
if not all([JIRA_EMAIL, JIRA_API_TOKEN, JIRA_DOMAIN]):
    raise ValueError("Missing required environment variables: JIRA_EMAIL, JIRA_API_TOKEN, or JIRA_DOMAIN")
JIRA_BASE_URL = f"https://{JIRA_DOMAIN}"

# FastAPI app initialization
app = FastAPI(
    title="JIRA Integration API",
    description="Fetch and manage JIRA Boards, Epics, Stories, Tasks with assignee information and reassignment"
)

# ---------------------- HELPERS ----------------------

def get_jira_session():
    return HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN), {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def safe_request(method, url, headers, auth, **kwargs):
    try:
        response = requests.request(method, url, headers=headers, auth=auth, **kwargs)
        print(f"[DEBUG] {method.upper()} {url} -> Status: {response.status_code}, Content: {response.text[:500]}")
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Authentication failed. Check JIRA_EMAIL and JIRA_API_TOKEN.")
        elif response.status_code == 403:
            raise HTTPException(status_code=403, detail="Permission denied. Check JIRA permissions.")
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Resource not found: {url}")
        elif response.status_code == 415:
            raise HTTPException(status_code=415, detail=f"Unsupported Media Type for {url}")
        response.raise_for_status()
        return response.json() if response.content else {}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during {method.upper()} request to {url}: {str(e)}")

def extract_description(description_field):
    if not description_field:
        return "No description"
    try:
        content = description_field.get("content", [])
        parts = []
        for c in content:
            if "content" in c:
                for inner in c["content"]:
                    if inner.get("type") == "text":
                        parts.append(inner.get("text", ""))
        return " ".join(parts).strip() or "No description"
    except Exception as e:
        print(f"[WARN] Error extracting description: {e}")
        return "No description"

def extract_comments(comment_field):
    if not comment_field or not comment_field.get("comments"):
        return []
    try:
        return [
            {
                "author": c["author"]["displayName"],
                "body": " ".join(
                    inner.get("text", "") for outer in c["body"].get("content", [])
                    for inner in outer.get("content", []) if inner.get("type") == "text"
                ).strip() or "No content",
                "created": c["created"]
            }
            for c in comment_field.get("comments", [])
        ]
    except Exception as e:
        print(f"[WARN] Error extracting comments: {e}")
        return []

def extract_assignee(assignee_field):
    if not assignee_field:
        return {"accountId": None, "displayName": "Unassigned"}
    try:
        return {
            "accountId": assignee_field.get("accountId", None),
            "displayName": assignee_field.get("displayName", "Unknown")
        }
    except Exception as e:
        print(f"[WARN] Error extracting assignee: {e}")
        return {"accountId": None, "displayName": "Unassigned"}

def search_issues(jql: str, max_results: int = 50):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/search"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,subtasks,description,comment,attachment,assignee,issuetype"
    }
    return safe_request("GET", url, headers, auth, params=params).get("issues", [])

_epic_link_field_id_cache = None

def get_epic_link_field_id():
    global _epic_link_field_id_cache
    if _epic_link_field_id_cache:
        return _epic_link_field_id_cache

    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/field"
    fields = safe_request("GET", url, headers, auth)

    for field in fields:
        if field.get("name", "").lower() == "epic link":
            _epic_link_field_id_cache = field["id"]
            return _epic_link_field_id_cache

    print("[WARN] 'Epic Link' field not found. Falling back to parent field.")
    return None

def fetch_stories(epic_key: str):
    jqls = [f'parent = "{epic_key}"']
    epic_link_field = get_epic_link_field_id()
    if epic_link_field:
        jqls.append(f'"{epic_link_field}" = "{epic_key}"')

    stories = []
    for jql in jqls:
        try:
            issues = search_issues(jql)
            if issues:
                stories = issues
                break
        except Exception as e:
            print(f"[WARN] Failed to fetch stories with JQL '{jql}': {e}")

    return {
        "stories": stories,
        "epic_key": epic_key
    }
def verify_task(task_id: str) -> bool:
    """
    Checks if a JIRA issue (task) exists.
    Returns True if the issue is found, False if not.
    """
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}"
    try:
        response = safe_request("GET", url, headers=headers, auth=auth)
        return response.get("id") is not None
    except HTTPException as e:
        if e.status_code == 404:
            return False
        raise

# ---------------------- MODELS ----------------------

class AssigneeModel(BaseModel):
    accountId: Optional[str]
    displayName: str

class SubtaskModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    assignee: AssigneeModel
    comments: List[Any] = []
    attachments: List[Any] = []

class TaskModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    assignee: AssigneeModel
    comments: List[Any] = []
    attachments: List[Any] = []
    subtasks: List[SubtaskModel] = []

class StoryModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    assignee: AssigneeModel
    comments: List[Any] = []
    attachments: List[Any] = []
    tasks: List[TaskModel]

class EpicModel(BaseModel):
    id: str
    key: str
    name: str
    description: Optional[str] = None
    assignee: AssigneeModel
    comments: List[Any] = []
    attachments: List[Any] = []
    stories: List[StoryModel]

class BoardModel(BaseModel):
    id: int
    name: str
    epics: List[EpicModel]

class ReassignIssueRequest(BaseModel):
    task_summary: str
    assignee_account_id: Optional[str]  # Allow null for unassigning
    
class CloseTaskRequest(BaseModel):
    project_id: Optional[str] = None
    epic_id: Optional[str] = None
    task_id: str
    comment: Optional[str] = None


class IssueInput(BaseModel):
    id: str
    task: str
    description: str
    link: Optional[str] = Field(None, description="Optional URL (e.g., https://example.com)")

    @validator("link", pre=True)
    @classmethod
    def validate_link(cls, v):
        if v and not v.startswith(("http://", "https://")):
            return "https://" + v
        return v

# ---------------------- READ ENDPOINTS ----------------------

@app.post("/create_issue", summary="Create a Jira issue from form input and push to backlog")
async def create_issue(
    id: str = Form(...),
    task: str = Form(...),
    description: str = Form(...),
    link: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    project_key: str = Form(...),
    issue_type: str = Form("Task")
):
    try:
        if not id.strip() or len(id) > 100:
            raise HTTPException(status_code=400, detail="Invalid ID")
        if not task.strip() or len(task) > 200:
            raise HTTPException(status_code=400, detail="Invalid task")
        if not description.strip() or len(description) > 1000:
            raise HTTPException(status_code=400, detail="Invalid description")

        image_content = None
        if image:
            allowed_types = ["image/jpeg", "image/png", "image/gif"]
            if image.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Invalid file type")
            max_size = 5 * 1024 * 1024
            image_content = await image.read()
            if len(image_content) > max_size:
                raise HTTPException(status_code=400, detail="File too large")

        if link and not link.startswith(("http://", "https://")):
            link = "https://" + link

        # Combine description and clickable hyperlink
        full_description = [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": description}]
            }
        ]
        if link:
            full_description.append({
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Click here: ", "marks": []},
                    {
                        "type": "text",
                        "text": link,
                        "marks": [{"type": "link", "attrs": {"href": link}}]
                    }
                ]
            })

        # Step 1: Create the Jira issue
        auth, base_headers = get_jira_session()
        issue_url = f"{JIRA_BASE_URL}/rest/api/3/issue"
        issue_payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": task,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": full_description
                },
                "issuetype": {"name": issue_type}
            }
        }

        print(json.dumps(issue_payload, indent=2))
        issue_response = safe_request("POST", issue_url, base_headers, auth, json=issue_payload)
        issue_key = issue_response.get("key")

        # Step 2: Upload attachments using a clean headers object
        attach_headers = {
            "Accept": "application/json",
            "X-Atlassian-Token": "no-check"
        }
        upload_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/attachments"
        files = []

        if image_content:
            files.append(('file', (image.filename, image_content, image.content_type)))

        uploaded_attachments = []
        if files:
            try:
                print(f"[DEBUG] Uploading attachments to {upload_url}...")
                response = requests.post(upload_url, headers=attach_headers, auth=auth, files=files)
                print(f"[DEBUG] Upload status: {response.status_code}, Content: {response.text}")
                response.raise_for_status()
                uploaded_attachments = response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Attachment upload failed: {str(e)}")

        return {
            "message": "Issue created successfully",
            "issue_key": issue_key,
            "issue_url": f"{JIRA_BASE_URL}/browse/{issue_key}",
            "attachments": uploaded_attachments
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Jira issue: {str(e)}")


@app.get("/boards", summary="Fetch all boards")
def fetch_boards():
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board"
    return safe_request("GET", url, headers, auth).get("values", [])

@app.get("/boards/{board_id}/epics", summary="Fetch epics for a board with additional metadata")
def fetch_epics(board_id: int):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board/{board_id}/epic"
    epics = safe_request("GET", url, headers, auth, params={"fields": "summary,description,comment,attachment,assignee"}).get("values", [])

    epics_with_metadata = []
    for epic in epics:
        epic_key = epic.get("key")
        print(f"[INFO] Processing epic: {epic_key}")

        try:
            epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment,assignee"
            epic_data = safe_request("GET", epic_url, headers, auth)
            fields = epic_data.get("fields", {})
            description = extract_description(fields.get("description", {}))
            comments = extract_comments(fields.get("comment", {}))
            assignee = extract_assignee(fields.get("assignee", {}))
            attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in fields.get("attachment", [])
            ]
        except Exception as e:
            description = f"Error fetching description: {str(e)}"
            comments = f"Error fetching comments: {str(e)}"
            assignee = {"accountId": None, "displayName": f"Error fetching assignee: {str(e)}"}
            attachments = f"Error fetching attachments: {str(e)}"

        epic_with_metadata = epic.copy()
        epic_with_metadata.update({
            "description": description,
            "comments": comments,
            "assignee": assignee,
            "attachments": attachments
        })
        epics_with_metadata.append(epic_with_metadata)

    return epics_with_metadata

@app.get("/epics/{epic_key}/stories", summary="Fetch stories and metadata for an epic")
def fetch_epic_details_with_stories(epic_key: str):
    auth, headers = get_jira_session()
    jqls = [f'parent = "{epic_key}"']
    epic_link_field = get_epic_link_field_id()
    if epic_link_field:
        jqls.append(f'"{epic_link_field}" = "{epic_key}"')

    stories = []
    for jql in jqls:
        issues = search_issues(jql)
        if issues:
            stories = issues
            break

    if not stories:
        raise HTTPException(
            status_code=404,
            detail=f"No stories found for epic '{epic_key}'. Tried JQLs: {jqls}"
        )

    try:
        epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment,assignee"
        epic_data = safe_request("GET", epic_url, headers, auth)
        fields = epic_data.get("fields", {})
        description = extract_description(fields.get("description", {}))
        comments = extract_comments(fields.get("comment", {}))
        assignee = extract_assignee(fields.get("assignee", {}))
        attachments = [
            {"filename": a["filename"], "content": a["content"], "created": a["created"]}
            for a in fields.get("attachment", [])
        ]
    except Exception as e:
        description = f"Error fetching description: {str(e)}"
        comments = f"Error fetching comments: {str(e)}"
        assignee = {"accountId": None, "displayName": f"Error fetching assignee: {str(e)}"}
        attachments = f"Error fetching attachments: {str(e)}"

    return {
        "epic_key": epic_key,
        "stories": stories,
        "description": description,
        "comments": comments,
        "assignee": assignee,
        "attachments": attachments
    }

@app.get("/stories/{story_key}/tasks", summary="Fetch tasks and subtasks linked to a story")
def fetch_tasks_and_subtasks(story_key: str):
    auth, headers = get_jira_session()
    issue_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{story_key}?fields=summary,description,comment,attachment,issuelinks,subtasks,assignee"
    issue_data = safe_request("GET", issue_url, headers, auth)

    issue_links = issue_data.get("fields", {}).get("issuelinks", [])
    print(f"[DEBUG] Issue links for {story_key}: {issue_links}")

    tasks = []
    for link in issue_links:
        link_type = link.get("type", {}).get("name", "").lower()
        inward = link.get("inwardIssue")
        outward = link.get("outwardIssue")

        linked_issue = inward if inward and inward.get("fields", {}).get("issuetype", {}).get("name") == "Task" else \
                       outward if outward and outward.get("fields", {}).get("issuetype", {}).get("name") == "Task" else None

        if linked_issue:
            issue_key = linked_issue["key"]
            issue_summary = linked_issue["fields"]["summary"]

            task_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=summary,description,comment,attachment,subtasks,assignee"
            task_data = safe_request("GET", task_url, headers, auth)
            fields = task_data.get("fields", {})

            subtasks_raw = fields.get("subtasks", [])
            subtasks = []
            for sub in subtasks_raw:
                sub_key = sub["key"]
                sub_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{sub_key}?fields=summary,description,comment,attachment,assignee"
                sub_data = safe_request("GET", sub_url, headers, auth)
                sub_fields = sub_data.get("fields", {})

                sub_description = extract_description(sub_fields.get("description", {}))
                sub_comments = extract_comments(sub_fields.get("comment", {}))
                sub_assignee = extract_assignee(sub_fields.get("assignee", {}))
                sub_attachments = [
                    {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                    for a in sub_fields.get("attachment", [])
                ]

                subtasks.append({
                    "id": sub["id"],
                    "key": sub["key"],
                    "summary": sub_fields.get("summary", "No summary"),
                    "description": sub_description,
                    "assignee": sub_assignee,
                    "comments": sub_comments,
                    "attachments": sub_attachments
                })

            description = extract_description(fields.get("description", {}))
            comments = extract_comments(fields.get("comment", {}))
            assignee = extract_assignee(fields.get("assignee", {}))
            attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in fields.get("attachment", [])
            ]

            task_info = {
                "id": task_data["id"],
                "key": issue_key,
                "summary": issue_summary,
                "description": description,
                "assignee": assignee,
                "comments": comments,
                "attachments": attachments,
                "subtasks": subtasks
            }
            tasks.append(task_info)

    print(f"[DEBUG] Final tasks list for {story_key}: {tasks}")
    return tasks

@app.get("/teams/project", summary="Get users in a project")
def get_users_in_project(project_key: str = Query(..., description="Project key to fetch users for")):
    auth, headers = get_jira_session()
    base_url = f"{JIRA_BASE_URL}/rest/api/3/project/{project_key}/role"

    try:
        roles = safe_request("GET", base_url, headers, auth)
        all_users = []
        for role_name, role_url in roles.items():
            role_data = safe_request("GET", role_url, headers, auth)
            actors = role_data.get("actors", [])

            for actor in actors:
                user_info = {
                    "role": role_name,
                    "displayName": actor.get("displayName"),
                    "accountId": actor.get("actorUser", {}).get("accountId"),
                    "email": actor.get("actorUser", {}).get("emailAddress")
                }
                all_users.append(user_info)

        if not all_users:
            raise HTTPException(status_code=404, detail=f"No users found for project '{project_key}'.")
        return all_users

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users for project: {str(e)}")

@app.get("/hierarchy", response_model=List[BoardModel], summary="Fetch full board-epic-story-task hierarchy")
def build_hierarchical_structure():
    auth, headers = get_jira_session()
    boards_data = []

    boards = fetch_boards()
    print(f"[INFO] Found {len(boards)} boards")

    for board in boards:
        print(f"[INFO] Processing board: {board['name']} (ID: {board['id']})")
        epics_data = []

        try:
            epics = fetch_epics(board["id"])
            print(f"[INFO]  -> Found {len(epics)} epics on board {board['id']}")
        except Exception as e:
            print(f"[WARN] Failed to fetch epics for board {board['id']}: {e}")
            continue

        for epic in epics:
            epic_key = epic.get("key")
            print(f"[INFO]   -> Processing epic: {epic_key}")

            epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment,assignee"
            epic_data = safe_request("GET", epic_url, headers, auth)
            fields = epic_data.get("fields", {})

            epic_description = extract_description(fields.get("description", {}))
            epic_comments = extract_comments(fields.get("comment", {}))
            epic_assignee = extract_assignee(fields.get("assignee", {}))
            epic_attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in fields.get("attachment", [])
            ]

            stories_data = []
            try:
                stories_response = fetch_stories(epic_key)
                stories = stories_response.get("stories", [])
                print(f"[INFO]     -> Found {len(stories)} stories for epic {epic_key}")

                for story in stories:
                    story_key = story["key"]
                    try:
                        story_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{story_key}?fields=summary,description,comment,attachment,assignee"
                        story_data = safe_request("GET", story_url, headers, auth)
                        story_fields = story_data.get("fields", {})

                        story_description = extract_description(story_fields.get("description", {}))
                        story_comments = extract_comments(story_fields.get("comment", {}))
                        story_assignee = extract_assignee(story_fields.get("assignee", {}))
                        story_attachments = [
                            {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                            for a in story_fields.get("attachment", [])
                        ]

                        tasks = fetch_tasks_and_subtasks(story_key)
                        task_models = [
                            TaskModel(
                                id=task["id"],
                                key=task["key"],
                                summary=task["summary"],
                                description=task.get("description", "No description"),
                                assignee=task.get("assignee", {"accountId": None, "displayName": "Unassigned"}),
                                comments=task.get("comments", []),
                                attachments=task.get("attachments", []),
                                subtasks=[
                                    SubtaskModel(
                                        id=sub["id"],
                                        key=sub["key"],
                                        summary=sub["summary"],
                                        description=sub.get("description", "No description"),
                                        assignee=sub.get("assignee", {"accountId": None, "displayName": "Unassigned"}),
                                        comments=sub.get("comments", []),
                                        attachments=sub.get("attachments", [])
                                    ) for sub in task.get("subtasks", [])
                                ]
                            )
                            for task in tasks
                        ]

                        stories_data.append(
                            StoryModel(
                                id=story["id"],
                                key=story_key,
                                summary=story_fields.get("summary", "No summary"),
                                description=story_description,
                                assignee=story_assignee,
                                comments=story_comments,
                                attachments=story_attachments,
                                tasks=task_models
                            )
                        )
                    except Exception as task_err:
                        print(f"[WARN]       -> Failed to fetch tasks for story {story_key}: {task_err}")

            except Exception as story_err:
                print(f"[WARN]     -> Failed to fetch stories for epic {epic_key}: {story_err}")

            epics_data.append(
                EpicModel(
                    id=str(epic.get("id", "unknown")),
                    key=epic_key,
                    name=epic.get("name") or epic.get("summary", "Unnamed Epic"),
                    description=epic_description,
                    assignee=epic_assignee,
                    comments=epic_comments,
                    attachments=epic_attachments,
                    stories=stories_data
                )
            )

        boards_data.append(
            BoardModel(
                id=board["id"],
                name=board["name"],
                epics=epics_data
            )
        )

    return boards_data

@app.get("/hierarchy/save", summary="Save hierarchy to a file")
def save_hierarchy_to_file():
    data = build_hierarchical_structure()
    file_path = "jira_hierarchy.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([d.dict() for d in data], f, indent=2, ensure_ascii=False)
    return {"status": "success", "message": f"Data saved to {file_path}"}

# ---------------------- ISSUE ENDPOINTS ----------------------

@app.get("/issues/{issue_key}/description", summary="Get issue description")
def get_issue_description(issue_key: str):
    issue_data = get_issue(issue_key)
    description = extract_description(issue_data.get("fields", {}).get("description", {}))
    return {"description": description}

@app.put("/issues/{issue_key}/description", summary="Update issue description")
def update_issue_description(issue_key: str, description: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    payload = {
        "fields": {
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            }
        }
    }
    return safe_request("PUT", url, headers, auth, json=payload)

@app.get("/issues/{issue_key}/comments", summary="List comments on an issue")
def list_issue_comments(issue_key: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment?fields=comments"
    comments_data = safe_request("GET", url, headers, auth)
    return extract_comments(comments_data)

@app.post("/issues/{issue_key}/comments", summary="Add comment to an issue")
def add_comment(issue_key: str, comment: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment"
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": comment}]
                }
            ]
        }
    }
    return safe_request("POST", url, headers, auth, json=payload)

@app.get("/issues/{issue_key}/attachments", summary="List attachments of an issue")
def list_attachments(issue_key: str):
    issue_data = get_issue(issue_key)
    attachments = [
        {"filename": a["filename"], "content": a["content"], "created": a["created"]}
        for a in issue_data.get("fields", {}).get("attachment", [])
    ]
    return attachments

@app.post("/issues/{issue_key}/attachments", summary="Add attachment to an issue")
async def add_attachment(issue_key: str, file: UploadFile = File(...)):
    auth, headers = get_jira_session()
    headers = {
        "X-Atlassian-Token": "no-check"
    }
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/attachments"

    try:
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        files = {'file': (file.filename, file_content, file.content_type or 'application/octet-stream')}
        print(f"[DEBUG] Uploading file: {file.filename}, Content-Type: {file.content_type}, Size: {len(file_content)} bytes")
        print(f"[DEBUG] Request headers: {headers}")
        response = requests.post(url, headers=headers, auth=auth, files=files)
        response.raise_for_status()
        print(f"[DEBUG] Response: {response.status_code}, {response.text[:500]}")
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error uploading attachment to {issue_key}: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error uploading attachment: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Failed to upload attachment to {issue_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading attachment: {str(e)}")

@app.post("/issues", summary="Create an issue")
def create_issue(project_key: str, summary: str, issue_type: str = "Task"):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type}
        }
    }
    return safe_request("POST", url, headers, auth, json=payload)

@app.get("/issues", summary="Get all issues in a project")
def list_issues(project_key: str = Query(..., description="Project key like 'KAN'")):
    jql = f"project = {project_key} ORDER BY created DESC"
    return search_issues(jql)

@app.put("/issues/{issue_id}", summary="Update an issue")
def update_issue(issue_id: str, summary: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_id}"
    payload = {"fields": {"summary": summary}}
    return safe_request("PUT", url, headers, auth, json=payload)

@app.delete("/issues/{issue_id}", summary="Delete an issue")
def delete_issue(issue_id: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_id}"
    try:
        response = requests.delete(url, headers=headers, auth=auth)
        response.raise_for_status()
        return {"detail": "Issue deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting issue: {str(e)}")

@app.get("/issues/{issue_key}", summary="Get issue details")
def get_issue(issue_key: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=summary,description,comment,attachment,assignee,issuetype"
    return safe_request("GET", url, headers, auth)

@app.post("/jira/api/{issue_key}/reassign_ticket", summary="Reassign an issue to a new user")
async def reassign_issue(issue_key: str, reassign_request: ReassignIssueRequest = Body(...)):
    auth, headers = get_jira_session()
    
    # Verify the issue exists
    issue_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=summary,issuetype,project"
    try:
        issue_data = safe_request("GET", issue_url, headers, auth)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"Issue {issue_key} not found or inaccessible: {str(e)}")

    # Verify the issue summary matches
    issue_summary = issue_data.get("fields", {}).get("summary", "")
    if reassign_request.task_summary != issue_summary:
        raise HTTPException(status_code=400, detail=f"Issue summary '{issue_summary}' does not match provided summary '{reassign_request.task_summary}'")

    # Verify the assignee is a valid user in the project (if assignee_account_id is provided)
    project_key = issue_data.get("fields", {}).get("project", {}).get("key")
    if reassign_request.assignee_account_id:
        users = get_users_in_project(project_key=project_key)
        valid_account_ids = [user["accountId"] for user in users if user["accountId"]]
        if reassign_request.assignee_account_id not in valid_account_ids:
            raise HTTPException(status_code=400, detail=f"Assignee account ID {reassign_request.assignee_account_id} is not a valid user in project '{project_key}'")

    # Reassign the issue
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    assignee = {"accountId": reassign_request.assignee_account_id} if reassign_request.assignee_account_id else None
    payload = {"fields": {"assignee": assignee}}
    try:
        response = safe_request("PUT", url, headers, auth, json=payload)
        assignee_name = "Unassigned" if not reassign_request.assignee_account_id else next(
            (user["displayName"] for user in get_users_in_project(project_key=project_key) if user["accountId"] == reassign_request.assignee_account_id),
            "Unknown"
        )
        return {
            "status": "success",
            "message": f"Issue {issue_key} reassigned to {assignee_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reassigning issue {issue_key}: {str(e)}")
@app.delete("/jira/api/{task_id}/closing_ticket", summary="Close a JIRA task")
async def close_ticket(
    task_id: str,
    request: CloseTaskRequest = Body(...)
):
    """
    Close a JIRA task by transitioning it to 'Done'.

    Parameters:
    - task_id (path): JIRA issue key or ID (e.g., SCRUM-123).
    - request (body): Optional JSON with a comment field.

    Returns:
    - JSON confirming the task was closed.

    Raises:
    - 404: If task is not found.
    - 400: If no 'Done' transition is available.
    - 500: On internal errors.
    """
    print(f"[DEBUG] URL task_id: {task_id}, Payload: {request}")

    # Step 0: Verify task exists
    if not verify_task(task_id):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    auth, headers = get_jira_session()

    # Step 1: Get transitions
    transition_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}/transitions"
    try:
        transitions_response = safe_request("GET", transition_url, headers, auth)
        transitions = transitions_response.get("transitions", [])
        done_transition = next(
            (t for t in transitions if t["to"]["name"].lower() == "done"),
            None
        )

        if not done_transition:
            raise HTTPException(status_code=400, detail="No 'Done' transition available for this issue")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transitions: {str(e)}")

    # Step 2: Add optional comment
    if request.comment:
        try:
            comment_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}/comment"
            comment_payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": request.comment}]
                        }
                    ]
                }
            }
            safe_request("POST", comment_url, headers, auth, json=comment_payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add comment: {str(e)}")

    # Step 3: Transition to Done
    try:
        transition_payload = {
            "transition": {"id": done_transition["id"]}
        }
        safe_request("POST", transition_url, headers, auth, json=transition_payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform transition: {str(e)}")

    return {
        "message": f"Task {task_id} successfully transitioned to 'Done'",
        "transition_id": done_transition["id"]
    }
