import json
import shutil
import tarfile
from pathlib import Path

import brainglobe_space as bgs
import meshio as mio
import numpy as np
import tifffile

import brainglobe_atlasapi.atlas_generation
from brainglobe_atlasapi import BrainGlobeAtlas, descriptors
from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    create_metadata_files,
    generate_metadata_dict,
)
from brainglobe_atlasapi.atlas_generation.stacks import (
    save_annotation,
    save_hemispheres,
    save_reference,
    save_secondary_reference,
)
from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
)
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    get_all_validation_functions,
)
from brainglobe_atlasapi.structure_tree_util import get_structures_tree
from brainglobe_atlasapi.utils import atlas_name_from_repr

# This should be changed every time we make changes in the atlas
# structure:
ATLAS_VERSION = brainglobe_atlasapi.atlas_generation.__version__


def filter_structures_not_present_in_annotation(structures, annotation):
    """
    Filter out structures that are not present in the annotation volume,
    or whose children are not present. Also prints removed structures.

    Args:
        structures (list of dict): List containing structure information
        annotation (np.ndarray): Annotation volume

    Returns:
        list of dict: Filtered list of structure dictionaries
    """
    present_ids = set(np.unique(annotation))
    # Create a structure tree for easy parent-child relationship traversal
    tree = get_structures_tree(structures)

    # Function to check if a structure or any of its descendants are present
    def is_present(structure_id):
        if structure_id in present_ids:
            return True
        # Recursively check all descendants
        for child_node in tree.children(structure_id):
            if is_present(child_node.identifier):
                return True
        return False

    removed = [s for s in structures if not is_present(s["id"])]
    for r in removed:
        print("Removed structure:", r["name"], "(ID:", r["id"], ")")

    return [s for s in structures if is_present(s["id"])]


def wrapup_atlas_from_data(
    atlas_name,
    atlas_minor_version,
    citation,
    atlas_link,
    species,
    resolution,
    orientation,
    root_id,
    reference_stack,
    annotation_stack,
    structures_list,
    meshes_dict,
    working_dir,
    atlas_packager=None,
    hemispheres_stack=None,
    cleanup_files=False,
    compress=True,
    scale_meshes=False,
    resolution_mapping=None,
    additional_references={},
    additional_metadata={},
):
    """
    Finalise an atlas with truly consistent format from all the data.

    Parameters
    ----------
    atlas_name : str
        Atlas name in the form author_species.
    atlas_minor_version : int or str
        Minor version number for this particular atlas.
    citation : str
        Citation for the atlas, if unpublished specify "unpublished".
    atlas_link : str
        Valid URL for the atlas.
    species : str
        Species name formatted as "CommonName (Genus species)".
    resolution : tuple
        Three elements tuple, resolution on three axes
    orientation :
        Orientation of the original atlas
        (tuple describing origin for BGSpace).
    root_id :
        Id of the root element of the atlas.
    reference_stack : str or Path or numpy array
        Reference stack for the atlas.
        If str or Path, will be read with tifffile.
    annotation_stack : str or Path or numpy array
        Annotation stack for the atlas.
        If str or Path, will be read with tifffile.
    structures_list : list of dict
        List of valid dictionary for structures.
    meshes_dict : dict
        dict of meshio-compatible mesh file paths in the form
        {sruct_id: meshpath}
    working_dir : str or Path obj
        Path where the atlas folder and compressed file will be generated.
    atlas_packager : str or None
        Credit for those responsible for converting the atlas
        into the BrainGlobe format.
    hemispheres_stack : str or Path or numpy array, optional
        Hemisphere stack for the atlas.
        If str or Path, will be read with tifffile.
        If none is provided, atlas is assumed to be symmetric.
    cleanup_files : bool, optional
         (Default value = False)
    compress : bool, optional
         (Default value = True)
    scale_meshes: bool, optional
        (Default values = False).
        If True the meshes points are scaled by the resolution
        to ensure that they are specified in microns,
        regardless of the atlas resolution.
    resolution_mapping: list, optional
        a list of three mapping the target space axes to the source axes
        only needed for mesh scaling of anisotropic atlases
    additional_references: dict, optional
        (Default value = empty dict).
        Dictionary with secondary reference stacks.
    additional_metadata: dict, optional
        (Default value = empty dict).
        Additional metadata to write to metadata.json
    """

    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = hemispheres_stack is None
    if isinstance(annotation_stack, str) or isinstance(annotation_stack, Path):
        annotation_stack = tifffile.imread(annotation_stack)
    structures_list = filter_structures_not_present_in_annotation(
        structures_list, annotation_stack
    )

    # Instantiate BGSpace obj, using original stack size in um as meshes
    # are un um:
    original_shape = reference_stack.shape
    volume_shape = tuple(res * s for res, s in zip(resolution, original_shape))
    space_convention = bgs.AnatomicalSpace(orientation, shape=volume_shape)

    # Check consistency of structures .json file:
    check_struct_consistency(structures_list)

    atlas_dir_name = atlas_name_from_repr(
        atlas_name, resolution[0], ATLAS_VERSION, atlas_minor_version
    )

    dest_dir = Path(working_dir) / atlas_dir_name

    # exist_ok would be more permissive but error-prone here as there might
    # be old files
    dest_dir.mkdir()

    stack_list = [reference_stack, annotation_stack]
    saving_fun_list = [save_reference, save_annotation]

    # If the atlas is not symmetric, we are also providing an hemisphere stack:
    if not symmetric:
        stack_list += [
            hemispheres_stack,
        ]
        saving_fun_list += [
            save_hemispheres,
        ]

    # write tiff stacks:
    for stack, saving_function in zip(stack_list, saving_fun_list):
        if isinstance(stack, str) or isinstance(stack, Path):
            stack = tifffile.imread(stack)

        # Reorient stacks if required:
        stack = space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        shape = stack.shape

        saving_function(stack, dest_dir)

    for k, stack in additional_references.items():
        stack = space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        save_secondary_reference(stack, k, output_dir=dest_dir)

    # Reorient vertices of the mesh.
    mesh_dest_dir = dest_dir / descriptors.MESHES_DIRNAME
    mesh_dest_dir.mkdir()

    for mesh_id, meshfile in meshes_dict.items():
        mesh = mio.read(meshfile)

        if scale_meshes:
            # Scale the mesh to the desired resolution, BEFORE transforming:
            # Note that this transformation happens in original space,
            # but the resolution is passed in target space (typically ASR)
            if not resolution_mapping:
                # isotropic case, so don't need to re-map resolution
                mesh.points *= resolution
            else:
                # resolution needs to be transformed back
                # to original space in anisotropic case
                original_resolution = (
                    resolution[resolution_mapping[0]],
                    resolution[resolution_mapping[1]],
                    resolution[resolution_mapping[2]],
                )
                mesh.points *= original_resolution

        # Reorient points:
        mesh.points = space_convention.map_points_to(
            descriptors.ATLAS_ORIENTATION, mesh.points
        )

        # Save in meshes dir:
        mio.write(mesh_dest_dir / f"{mesh_id}.obj", mesh)

    # save regions list json:
    with open(dest_dir / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures_list, f)

    # Finalize metadata dictionary:
    metadata_dict = generate_metadata_dict(
        name=atlas_name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,  # We expect input to be asr
        orientation=descriptors.ATLAS_ORIENTATION,  # Pass orientation "asr"
        version=f"{ATLAS_VERSION}.{atlas_minor_version}",
        shape=shape,
        additional_references=[k for k in additional_references.keys()],
        atlas_packager=atlas_packager,
    )

    # Create human readable .csv and .txt files:
    create_metadata_files(
        dest_dir,
        metadata_dict,
        structures_list,
        root_id,
        additional_metadata=additional_metadata,
    )

    atlas_name_for_validation = atlas_name_from_repr(atlas_name, resolution[0])

    # creating BrainGlobe object from local folder (working_dir)
    atlas_to_validate = BrainGlobeAtlas(
        atlas_name=atlas_name_for_validation,
        brainglobe_dir=working_dir,
        check_latest=False,
    )

    # Run validation functions
    print(f"Running atlas validation on {atlas_dir_name}")

    validation_results = {}

    for func in get_all_validation_functions():
        try:
            func(atlas_to_validate)
            validation_results[func.__name__] = "Pass"
        except AssertionError as e:
            validation_results[func.__name__] = f"Fail: {str(e)}"

    def _check_validations(validation_results):
        # Helper function to check if all validations passed
        all_passed = all(
            result == "Pass" for result in validation_results.values()
        )

        if all_passed:
            print("This atlas is valid")
        else:
            failed_functions = [
                func
                for func, result in validation_results.items()
                if result != "Pass"
            ]
            error_messages = [
                result.split(": ")[1]
                for result in validation_results.values()
                if result != "Pass"
            ]

            print("These validation functions have failed:")
            for func, error in zip(failed_functions, error_messages):
                print(f"- {func}: {error}")

    _check_validations(validation_results)

    # Compress if required:
    if compress:
        output_filename = dest_dir.parent / f"{dest_dir.name}.tar.gz"
        print(f"Saving compressed atlas data at: {output_filename}")
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(dest_dir, arcname=dest_dir.name)

    # Cleanup if required:
    if cleanup_files:
        print(f"Cleaning up atlas data at: {dest_dir}")
        # Clean temporary directory and remove it:
        shutil.rmtree(dest_dir)

    return output_filename
