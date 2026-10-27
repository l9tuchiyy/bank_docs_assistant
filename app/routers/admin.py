"""Минимальная админка: загрузка материалов в RAG-базу без правки кода."""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from app.config import ADMIN_TOKEN, ADMIN_UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from app.deps import engine, kb

router = APIRouter(prefix="/api/admin", tags=["Администрирование"])


def _auth(token: str | None) -> None:
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(401, "Неверный токен администратора")


@router.get("/kb")
def kb_info(x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    return kb.info()


@router.post("/upload")
async def upload_knowledge_file(file: UploadFile = File(...), x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if not name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Разрешены PDF, DOCX, TXT и MD")
    data = await file.read()
    if not data or len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, "Файл пустой или превышает 10 МБ")
    ADMIN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = ADMIN_UPLOAD_DIR / f"{uuid4().hex}_{name}"
    target.write_bytes(data)
    count = kb.reload()
    engine.templates = engine.templates  # Явно сохраняем активные шаблоны: догрузка меняет только базу.
    return {"status": "uploaded", "filename": name, "chunks": count}


@router.post("/reload")
def reload_knowledge(x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    return {"chunks": kb.reload()}


def _display_name(stored: str) -> str:
    """Файлы лежат как <uuid hex>_<имя>; пользователю показываем только исходное имя."""
    head, sep, tail = stored.partition("_")
    return tail if sep and len(head) == 32 else stored


@router.get("/files", summary="Список файлов, загруженных через админку")
def list_admin_files(x_admin_token: str | None = Header(default=None)):
    _auth(x_admin_token)
    if not ADMIN_UPLOAD_DIR.exists():
        return {"files": []}
    files = [
        {"id": f.name, "name": _display_name(f.name), "size": f.stat().st_size, "uploaded_at": int(f.stat().st_mtime)}
        for f in sorted(ADMIN_UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if f.is_file() and not f.name.startswith(".")
    ]
    return {"files": files}


@router.delete("/files/{file_id}", summary="Удалить файл, загруженный через админку, и обновить базу")
def delete_admin_file(file_id: str, x_admin_token: str | None = Header(default=None)):
    """Удаляет только файлы из admin_uploads. Исходные материалы базы знаний этим методом удалить нельзя."""
    _auth(x_admin_token)
    target = (ADMIN_UPLOAD_DIR / Path(file_id).name).resolve()      # Path(...).name отсекает ../ и подпапки
    if target.parent != ADMIN_UPLOAD_DIR.resolve() or not target.is_file():
        raise HTTPException(404, "Файл не найден")
    target.unlink()
    return {"status": "deleted", "filename": _display_name(target.name), "chunks": kb.reload()}
