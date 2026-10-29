// batch cellpose-sam
// first imageJ macro
inputDir = getDirectory("Choose input image folder");
outDir = getDirectory("Choose ouput mask folder");


files = getFileList(inputDir);
saved = 0;

for(i = 0; i < files.length; i ++){
	name = files[i];
	path = inputDir + name;
	
	dot = lastIndexOf(name, ".");
	base = substring(name, 0, dot);
	output_Path = outDir + base + "_cp_mask.tif";
	
	print(name);
	//print(dot);
	print(base);
	print(output_Path);
	
	if(File.exists(output_Path)){
		print("Skipped exiting output");
	}else{
		open(path);
		inputID = getImageID();
	
		run("Cellpose-SAM...", "cp_model=cpsam_v2 custom_model= cell_diameter=40 chan0=2 chan1=1 chan2=None min_size=15 normalize=true resample=false return_rois=false cellprob_threshold=0.0 flow_threshold=0.5 tile_overlap=0.1 niter=2000 compute_flows=false shuffle=false labeltype=32-bit mode_3d=None stitch_threshold=0.0 flow3d_smooth=0 torchversion=cu126 usegpu=true");
	
		//guard  against accidentally saving the original image
		if(getImageID() == inputID)
			exit("No active ouput image for : " + names + "\nCheck the Cellpoes log." );
	
	
		//selectImage("11d_LNP3_20x_limb1-AiryStitch-01_8bit.tif");
	
		saveAs("Tiff", output_Path);
		close();
	
		selectImage(inputID);
		close();
	
		saved++;
	
		print("Saved: " + output_Path);
	
	
		//selectImage("11d_LNP3_20x_limb1-AiryStitch-01_8bit.tif_Cellpose-SAM");
	}	
}
