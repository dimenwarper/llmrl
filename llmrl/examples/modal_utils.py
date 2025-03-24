import tomli
import modal

def create_modal_image_from_pyproject(pyproject_path="pyproject.toml"):
    """
    Parse a pyproject.toml file and create a Modal image with the dependencies.
    
    Args:
        pyproject_path (str): Path to the pyproject.toml file
        
    Returns:
        tuple: (modal.Image, modal.App) The configured Modal image and app
    """
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
    
    project_info = pyproject_data.get("project", {})
    project_name = project_info.get("name", "default_app")
    dependencies = project_info.get("dependencies", [])
    
    app = modal.App(project_name)
    
    image = modal.Image.debian_slim()
    
    if dependencies:
        image = image.pip_install(*dependencies)
    
    print(f"Created Modal image for '{project_name}' with {len(dependencies)} dependencies")
    
    return image, app

IMAGE, APP = create_modal_image_from_pyproject()

@APP.local_entrypoint()
def maybe_run_with_modal(
        fun, 
        flag
):
    wrapped = APP.function(gpu="any", image=IMAGE)
    if flag:
        result = wrapped.remote()
    else:
        result = fun()
    return fun