from uuid import uuid4

task_status = {}

def create_task():
    task_id = str(uuid4())
    task_status[task_id] = "PENDING"
    return task_id

def update_task(task_id, status):
    task_status[task_id] = status

def get_status(task_id):
    return task_status.get(task_id, "NOT_FOUND")
