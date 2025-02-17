Detailed Implementation Plan for Fixing Docker Image Building and Pulling Issues in SWE‐bench Evaluation
==============================================================================================

Overview:
---------
The goal is to ensure that the evaluation harness builds all Docker images locally rather than attempting to pull them from Docker Hub. Currently, image tags (especially the instance images) are constructed without a local prefix, so Docker attempts to pull “swebench/…” images from a remote repository. We will update the naming conventions to include a “local/” prefix across the core modules and verify that all modules use the correct image tag.

Implementation Steps:
---------------------

1. Update Image Naming in TestSpec:
   - In the file “SWE-bench/swebench/harness/test_spec.py”, modify the computed image tag properties so that all image keys include a “local/” prefix.
   - Change the following properties:
     • base_image_key:  
       Before:  
         return f"sweb.base.{self.arch}:latest"  
       After:  
         return f"local/sweb.base.{self.arch}:latest"
     • env_image_key:  
       Before:  
         return f"sweb.env.{self.arch}.{val}:latest"  
       After:  
         return f"local/sweb.env.{self.arch}.{val}:latest"
     • instance_image_key:  
       Before:  
         return f"sweb.eval.{self.arch}.{self.instance_id}:latest"  
       After:  
         return f"local/sweb.eval.{self.arch}.{self.instance_id}:latest"
       
   Example snippet:
     --------------------------------
     @property
     def instance_image_key(self):
         return f"local/sweb.eval.{self.arch}.{self.instance_id}:latest"
     --------------------------------

2. Verify Docker Build Flow:
   - In “SWE-bench/swebench/harness/docker_build.py”, the build process uses the image tag from TestSpec (e.g., in build_instance_image and build_container).
   - Confirm that the methods pass the tag returned from TestSpec (which now starts with "local/") into the Docker API.
   - Optionally, add logging (or assert statements during development) to print out “test_spec.instance_image_key” right before it is used by client.api.build.

3. Check Dockerfile Generation:
   - In “SWE-bench/swebench/harness/dockerfiles.py” the instance Dockerfile is generated using get_dockerfile_instance(). Note that the Dockerfile uses a base image specified by “env_image_name” (from TestSpec).
   - Verify that since TestSpec.env_image_key is now prefixed with “local/”, the FROM clause in the generated Dockerfile will correctly reference a locally built image.
   - Sample call in get_dockerfile_instance:
         return _DOCKERFILE_INSTANCE.format(platform=platform, env_image_name=env_image_name)
   - No further changes should be needed here if TestSpec returns the right value.

4. Update run_evaluation.py:
   - In “SWE-bench/swebench/harness/run_evaluation.py”, container creation uses test_spec.instance_image_key as the image name.
   - Ensure that this value (and any other references) is now produced with the “local/” prefix.
   - No extra change is needed unless a hard-coded “swebench/…” is found elsewhere in this module.

5. Review Other Modules:
   - Check any related modules (for example, in swebench_docker such as dockerfile_generator.py) for image naming inconsistencies.
   - If those modules construct images using different conventions (e.g. using the namespace) then decide if they should be aligned with the local policy or left separate.
   - Document that the evaluation harness images must use the “local/” prefix so that they are built locally.

6. Testing & Verification:
   - After implementing the changes, clean up any stale images (e.g., run “docker image prune -f”).
   - Execute the evaluation via the report script (e.g., “python report.py predictions/results/”) and watch the logs.
   - Verify that Docker no longer attempts to pull images from a remote registry (i.e. no “pull access denied” errors) and that image names like “local/sweb.eval.arm64.&lt;instance_id&gt;:latest” are built and used.
   - Check log files for any remaining references to the old naming scheme.

This plan details the necessary code modifications and coordination across test_spec.py, docker_build.py, run_evaluation.py, and related modules to enforce local Docker image builds and resolve the 404 pull errors.
