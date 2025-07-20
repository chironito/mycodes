import falcon # falcon==4.0.2
from pathlib import Path
import hashlib
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not found."
import os
import uuid
import shutil
from io import BytesIO
from zipfile import ZipFile
import mimetypes

STREAM_THRESHOLD = 1024 * 10
MAX_UPLOAD_SIZE = 1024 * 1024 * 100

class FileList:
    def on_get(self, req, resp):
        key = req.get_header("Authorization", required=True)
        if hashlib.sha256(key.encode()).hexdigest() != os.environ['file_server_password_sha256']:
            resp.status = falcon.HTTP_401
            resp.text = "Unauthorized"
            return
        folder_path = req.get_param("folder_path", default=".")
        for _, dirnames, filenames in os.walk(folder_path):
            break
        resp.status = falcon.HTTP_200
        resp.media = {"folders": dirnames, "filenames": filenames}
        return

class GetObject:
    def on_get(self, req, resp, path):
        key = req.get_header("Authorization", required=True)
        if hashlib.sha256(key.encode()).hexdigest() != os.environ['file_server_password_sha256']:
            resp.status = falcon.HTTP_401
            resp.text = "Unauthorized"
            return
        stream = req.get_param_as_bool("stream", default=True)
        try:
            filesize = os.path.getsize(path)
            if filesize >= STREAM_THRESHOLD:
                resp.stream = open(path, "rb")
                resp.content_length = filesize
                resp.stream_len = filesize
            else:
                with open(path, "rb") as f:
                    content = f.read()
                resp.data = content
            resp.set_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            resp.content_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            resp.status = falcon.HTTP_200
        except IsADirectoryError:
            *filepaths, = Path(path).rglob("*")
            folder_size = sum(map(os.path.getsize, filepaths))
            if folder_size >= STREAM_THRESHOLD:
                zip_bytes_io = BytesIO()
                with ZipFile(zip_bytes_io, "w") as zipfile:
                    for filepath in filepaths:
                        zipfile.write(filepath)
                zip_bytes_io.seek(0, 2)
                filesize = zip_bytes_io.tell()
                zip_bytes_io.seek(0)
                resp.stream = zip_bytes_io
            else:
                with ZipFile(f"{path}.zip", "w") as zipfile:
                    for filepath in filepaths:
                        zipfile.write(filepath)
                filesize = os.path.getsize(f"{path}.zip")
                resp.stream = open(f"{path}.zip", "rb")
                try:
                    os.remove(f"{path}.zip")
                except FileNotFoundError:
                    pass
            resp.content_type = 'application/octet-stream'
            resp.content_length = filesize
            resp.set_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}.zip"')
            resp.stream_len = filesize
            resp.status = falcon.HTTP_200
        except (FileNotFoundError, TypeError):
            resp.status = falcon.HTTP_404
            resp.text = "File not found"
        finally:
            return
            
class CreateObject:
    def on_post(self, req, resp):
        key = req.get_header("Authorization", required=True)
        if hashlib.sha256(key.encode()).hexdigest() != os.environ['file_server_password_sha256']:
            resp.status = falcon.HTTP_401
            resp.text = "Unauthorized"
            return
        object_type = req.get_param("type", default="file")
        if object_type == "file":
            form = req.get_media()
            file_data = b""
            for part in form:
                if part.name == "file":
                    file_data = part.stream.read()
            if not file_data:
                resp.text = "No file data"
                resp.status = falcon.HTTP_422
                return
            filepath = req.get_param("filepath", f"./{uuid.uuid4()}")
            overwrite = req.get_param_as_bool("overwrite", default=False)
            try:
                if not overwrite and os.path.isfile(filepath):
                    resp.status = falcon.HTTP_409
                    resp.text = "File exists"
                    return
                with open(filepath, "wb") as f:
                    f.write(filepath)
                resp.text = filepath
                resp.status = falcon.HTTP_201
                return
            except FileNotFoundError:
                resp.text = "Invalid filepath"
                resp.status = falcon.HTTP_404
                return
        elif object_type == "folder":
            folder_path = req.get_param("folder_path", f"./{uuid.uuid4()}")
            try:
                os.mkdir(folder_path)
            except FileNotFoundError:
                resp.text = "Invalid folder path"
                resp.status = falcon.HTTP_404
                return
            except FileExistsError:
                resp.text = "Folder Exists"
                resp.status = falcon.HTTP_409
                return
            except Exception as exp:
                resp.text = str(exp)
                resp.status = falcon.HTTP_424
                return
            resp.status = falcon.HTTP_201
            resp.text = folder_path
            return
        else:
            resp.text = "Invalid object type"
            resp.status = falcon.HTTP_400
            return

class DeleteObject:
    def on_delete(self, req, resp, path):
        key = req.get_header("Authorization", required=True)
        if hashlib.sha256(key.encode()).hexdigest() != os.environ['file_server_password_sha256']:
            resp.status = falcon.HTTP_401
            resp.text = "Unauthorized"
            return
        object_path = path
        try:
            os.remove(object_path)
        except FileNotFoundError:
            resp.text = "File not Found"
            resp.status = falcon.HTTP_404
            return
        except IsADirectoryError:
            try:
                shutil.rmtree(object_path)
            except Exception as exp:
                resp.text = str(exp)
                resp.status = falcon.HTTP_424
                return
            else:
                resp.text = "Folder deleted"
                resp.status = falcon.HTTP_202
                return
        except TypeError:
            resp.text = "No Object path"
            resp.status = falcon.HTTP_404
            return
        else:
            resp.text = "File deleted"
            resp.status = falcon.HTTP_202
            return

app = falcon.App()

handler = falcon.media.MultipartFormHandler()
handler.parse_options.max_body_part_buffer_size = MAX_UPLOAD_SIZE
app.req_options.media_handlers[falcon.MEDIA_MULTIPART] = handler

app.add_route("/file_list", FileList())
app.add_route("/get_object/{path:path}", GetObject())
app.add_route("/create_object", CreateObject())
app.add_route("/delete_object/{path:path}", DeleteObject())
