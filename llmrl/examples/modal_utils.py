import tomli
import modal
from typing import Union
from pathlib import PurePosixPath
import inspect
from inspect import Parameter, Signature
import types


def create_function_with_dynamic_signature(param_specs, body_func, return_type=None):
    """
    util that creates a function with dynamic signature, mainly to dynamically generate modal entrypoints
    
    param_specs: List of tuples (param_name, param_type, default_value)
                 default_value is optional and can be inspect.Parameter.empty
    body_func: The function that will be called with the bound parameters
    return_type: Optional return type annotation
    """
    params = []
    
    for spec in param_specs:
        if len(spec) == 2:  # Just name and type
            name, annotation = spec
            default = Parameter.empty
        else:  # Name, type, and default value
            name, annotation, default = spec
            
        params.append(
            Parameter(
                name, 
                Parameter.POSITIONAL_OR_KEYWORD, 
                default=default,
                annotation=annotation
            )
        )
    
    sig = Signature(params, return_annotation=return_type)
    
    def template_func(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        return body_func(**bound_args.arguments)
    
    template_func.__signature__ = sig
    template_func.__annotations__ = {p.name: p.annotation for p in params if p.annotation is not Parameter.empty}
    
    if return_type is not Parameter.empty and return_type is not None:
        template_func.__annotations__['return'] = return_type
    
    return template_func


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

    image = (
        image
        .env(
            dict(
                HUGGINGFACE_HUB_CACHE="/pretrained",
                HF_HUB_ENABLE_HF_TRANSFER="1",
                TQDM_DISABLE="true",
            )
        )
        .entrypoint([])
    )
    
    print(f"Created Modal image for '{project_name}' with {len(dependencies)} dependencies")
    
    return image, app

image, app = create_modal_image_from_pyproject()

# Volumes for models/data 
artifact_volume = modal.Volume.from_name(
    "artifact-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/artifacts/": artifact_volume,
}


def test(args):
    def bla(**kwargs):
        print(kwargs)
    bla_with_sig = create_function_with_dynamic_signature(args, bla)
    return bla_with_sig

def maybe_run_with_modal(
        fun, 
        flag
):
    wrapped = app.function(gpu="any", image=image, volumes=VOLUME_CONFIG)(fun)
    if flag:
        result = wrapped.remote()
    else:
        result = fun()
    return result 