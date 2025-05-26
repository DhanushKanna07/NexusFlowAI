import os
import json
from fastapi import File, UploadFile, FastAPI, HTTPException, Query, APIRouter
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
from typing import List, Optional, Any
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()
router = APIRouter()

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
    description="Fetch and manage JIRA Boards, Epics, Stories, Tasks"
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

def search_issues(jql: str, max_results: int = 50):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/search"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,subtasks,description,comment,attachment"
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
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}"
    try:
        response = safe_request("GET", url, headers=headers, auth=auth)
        return response.get("id") is not None
    except HTTPException as e:
        if e.status_code == 404:
            return False
        raise

def close_task(task_id: str, comment: Optional[str] = None) -> dict:
    auth, headers = get_jira_session()
    
    # Get available transitions
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}/transitions"
    response = safe_request("GET", url, headers=headers, auth=auth)
    transitions = response.get("transitions", [])
    
    # Find "Done" or "Closed" transition
    done_transition = next(
        (t for t in transitions if t["name"].lower() in ["done", "closed"]),
        None
    )
    if not done_transition:
        raise HTTPException(
            status_code=400,
            detail="No 'Done' or 'Closed' transition available for this issue"
        )
    
    # Add comment if provided
    if comment:
        comment_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{task_id}/comment"
        comment_payload = {
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
        safe_request("POST", comment_url, headers=headers, auth=auth, json=comment_payload)
    
    # Perform the transition
    transition_payload = {
        "transition": {
            "id": done_transition["id"]
        }
    }
    response = safe_request("POST", url, headers=headers, auth=auth, json=transition_payload)
    return {"message": f"Task {task_id} closed successfully"}

# ---------------------- MODELS ----------------------

class SubtaskModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    comments: List[Any] = []
    attachments: List[Any] = []

class TaskModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    comments: List[Any] = []
    attachments: List[Any] = []
    subtasks: List[SubtaskModel] = []

class StoryModel(BaseModel):
    id: str
    key: str
    summary: str
    description: Optional[str] = None
    comments: List[Any] = []
    attachments: List[Any] = []
    tasks: List[TaskModel]

class EpicModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    comments: List[Any] = []
    attachments: List[Any] = []
    stories: List[StoryModel]

class BoardModel(BaseModel):
    id: int
    name: str
    epics: List[EpicModel]

class CloseTaskRequest(BaseModel):
    project_id: Optional[str] = None
    epic_id: Optional[str] = None
    task_id: str
    comment: Optional[str] = None

# ---------------------- READ ENDPOINTS ----------------------

@app.get("/boards", summary="Fetch all boards")
def fetch_boards():
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board"
    return safe_request("GET", url, headers, auth).get("values", [])

@app.get("/boards/{board_id}/epics", summary="Fetch epics for a board with additional metadata")
def fetch_epics(board_id: int):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board/{board_id}/epic"
    epics = safe_request("GET", url, headers, auth, params={"fields": "summary,description,comment,attachment"}).get("values", [])

    epics_with_metadata = []
    for epic in epics:
        epic_key = epic.get("key")
        print(f"[INFO] Processing epic: {epic_key}")

        try:
            epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment"
            epic_data = safe_request("GET", epic_url, headers, auth)
            description = extract_description(epic_data.get("fields", {}).get("description", {}))
            comments = extract_comments(epic_data.get("fields", {}).get("comment", {}))
            attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in epic_data.get("fields", {}).get("attachment", [])
            ]
        except Exception as e:
            description = f"Error fetching description: {str(e)}"
            comments = f"Error fetching comments: {str(e)}"
            attachments = f"Error fetching attachments: {str(e)}"

        epic_with_metadata = epic.copy()
        epic_with_metadata.update({
            "description": description,
            "comments": comments,
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
        epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment"
        epic_data = safe_request("GET", epic_url, headers, auth)
        description = extract_description(epic_data.get("fields", {}).get("description", {}))
        comments = extract_comments(epic_data.get("fields", {}).get("comment", {}))
        attachments = [
            {"filename": a["filename"], "content": a["content"], "created": a["created"]}
            for a in epic_data.get("fields", {}).get("attachment", [])
        ]
    except Exception as e:
        description = f"Error fetching description: {str(e)}"
        comments = f"Error fetching comments: {str(e)}"
        attachments = f"Error fetching attachments: {str(e)}"

    return {
        "epic_key": epic_key,
        "stories": stories,
        "description": description,
        "comments": comments,
        "attachments": attachments
    }

@app.get("/stories/{story_key}/tasks", summary="Fetch tasks and subtasks linked to a story")
def fetch_tasks_and_subtasks(story_key: str):
    auth, headers = get_jira_session()
    issue_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{story_key}?fields=summary,description,comment,attachment,issuelinks,subtasks"
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

            task_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=summary,description,comment,attachment,subtasks"
            task_data = safe_request("GET", task_url, headers, auth)
            fields = task_data.get("fields", {})

            subtasks_raw = fields.get("subtasks", [])
            subtasks = []
            for sub in subtasks_raw:
                sub_key = sub["key"]
                sub_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{sub_key}?fields=summary,description,comment,attachment"
                sub_data = safe_request("GET", sub_url, headers, auth)
                sub_fields = sub_data.get("fields", {})

                sub_description = extract_description(sub_fields.get("description", {}))
                sub_comments = extract_comments(sub_fields.get("comment", {}))
                sub_attachments = [
                    {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                    for a in sub_fields.get("attachment", [])
                ]

                subtasks.append({
                    "id": sub["id"],
                    "key": sub["key"],
                    "summary": sub_fields.get("summary", "No summary"),
                    "description": sub_description,
                    "comments": sub_comments,
                    "attachments": sub_attachments
                })

            description = extract_description(fields.get("description", {}))
            comments = extract_comments(fields.get("comment", {}))
            attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in fields.get("attachment", [])
            ]

            task_info = {
                "id": task_data["id"],
                "key": issue_key,
                "summary": issue_summary,
                "description": description,
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

            epic_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{epic_key}?fields=summary,description,comment,attachment"
            epic_data = safe_request("GET", epic_url, headers, auth)

            epic_description = extract_description(epic_data.get("fields", {}).get("description", {}))
            epic_comments = extract_comments(epic_data.get("fields", {}).get("comment", {}))
            epic_attachments = [
                {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                for a in epic_data.get("fields", {}).get("attachment", [])
            ]

            stories_data = []
            try:
                stories_response = fetch_stories(epic_key)
                stories = stories_response.get("stories", [])
                print(f"[INFO]     -> Found {len(stories)} stories for epic {epic_key}")

                for story in stories:
                    story_key = story["key"]
                    try:
                        story_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{story_key}?fields=summary,description,comment,attachment"
                        story_data = safe_request("GET", story_url, headers, auth)

                        story_description = extract_description(story_data.get("fields", {}).get("description", {}))
                        story_comments = extract_comments(story_data.get("fields", {}).get("comment", {}))
                        story_attachments = [
                            {"filename": a["filename"], "content": a["content"], "created": a["created"]}
                            for a in story_data.get("fields", {}).get("attachment", [])
                        ]

                        tasks = fetch_tasks_and_subtasks(story_key)
                        task_models = [
                            TaskModel(
                                id=task["id"],
                                key=task["key"],
                                summary=task["summary"],
                                description=task.get("description", "No description"),
                                comments=task.get("comments", []),
                                attachments=task.get("attachments", []),
                                subtasks=[
                                    SubtaskModel(
                                        id=sub["id"],
                                        key=sub["key"],
                                        summary=sub["summary"],
                                        description=sub.get("description", "No description"),
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
                                summary=story["fields"].get("summary", "No summary"),
                                description=story_description,
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
                    name=epic.get("name") or epic.get("summary", "Unnamed Epic"),
                    description=epic_description,
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
    headers["X-Atlassian-Token"] = "no-check"
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/attachments"

    try:
        file_content = await file.read()
        files = {'file': (file.filename, file_content, file.content_type)}
        response = requests.post(url, headers=headers, auth=auth, files=files)
        response.raise_for_status()
        return response.json()
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
    print(f"[DEBUG] Deleting issue: {issue_id}")
    response = safe_request("DELETE", url, headers=headers, auth=auth)
    return {"message": f"Issue {issue_id} deleted successfully"}

@app.get("/issues/{issue_key}", summary="Get issue details")
def get_issue(issue_key: str):
    auth, headers = get_jira_session()
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=summary,description,comment,attachment"
    return safe_request("GET", url, headers, auth)

@app.delete("/jira/api/{task_id}/closing_ticket", summary="Close a JIRA task")
async def close_ticket(task_id: str, request: CloseTaskRequest):
    """
    Close a JIRA task by transitioning it to 'Done' or 'Closed'.
    
    Parameters:
    - task_id (path): Issue key (e.g., SCRUM-52) or ID (e.g., 10203).
    - request (body): JSON payload with optional project_id, epic_id, task_id, and comment.
    
    Returns:
    - Success message if closed.
    
    Raises:
    - 404: If task not found.
    - 403: If insufficient permissions.
    """
    print(f"[DEBUG] URL task_id: {task_id}, Payload: {request}")
    if not verify_task(task_id):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return close_task(task_id, request.comment)