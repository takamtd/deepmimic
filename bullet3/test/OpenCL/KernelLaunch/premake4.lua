function createProject(vendor)
	
	hasCL = findOpenCL(vendor)
	
	if (hasCL) then

		project ("Test_OpenCL_kernel_launch_" .. vendor)

		initOpenCL(vendor)
	
		language "C++"
				
		
		kind "ConsoleApp"
		

		includedirs {"../../../src"}
		
		files {
			"main.cpp",
			"../../../src/Bullet3OpenCL/Initialize/b3OpenCLUtils.cpp",
			"../../../src/Bullet3Common/b3AlignedAllocator.cpp",
			"../../../src/Bullet3OpenCL/Initialize/b3OpenCLUtils.h",
			"../../../src/Bullet3Common/b3Logging.cpp",
			"../../../src/Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.cpp"
		}
		
	end
end
createProject("clew")	
createProject("Apple")
createProject("AMD")
createProject("Intel")
createProject("NVIDIA")
