# Uncompress outputs
for k in {1..8}; do 
	cd plate_"$k" && for j in *zip; do unzip $j& done && cd ../; 
done

# Count and report files
for k in *; do 
	n=$(ls $k/output/ | wc -l); 
	echo $k $n; 
done

# Find objects and pack results
for k in {1..8}; do
	python masks.py plate_"$k";
	cd plate_"$k"
	zip -r masks.zip masks && rm -Rf masks &
	zip -r prob-maps.zip prob-maps && rm -Rf prob-maps
       cd ../
done

# Upload results
for k in {1..8}; do
	cd plate_"$k"
	gcloud storage cp masks.zip gs://lasagna-emil/genome-wide_MN_screen/gws_MN_screen_processed_data/plate_"$k"/
	gcloud storage cp prob-maps.zip gs://lasagna-emil/genome-wide_MN_screen/gws_MN_screen_processed_data/plate_"$k"/
	cd ../
done

