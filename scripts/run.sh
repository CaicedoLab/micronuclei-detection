# Label objects and pack results
label_and_pack()
{
    P=$1
    python masks.py plate_"$P";
    cd plate_"$P"
    zip -r masks.zip masks && rm -Rf masks &
    zip -r prob-maps.zip prob-maps && rm -Rf prob-maps
    cd ../
}

# Uncompress outputs
for k in {1..8}; do 
	cd plate_"$k" && for j in *zip; do unzip $j& done && cd ../; 
	wait
done

for k in {1..8}; do
	label_and_pack $k &
done
wait

# Count and report files
for k in plate_*; do 
	n=$(ls $k/output/ | wc -l); 
	echo $k $n; 
done

# Upload results
for k in {1..8}; do
	cd plate_"$k"
	gcloud storage cp masks.zip gs://lasagna-emil/genome-wide_MN_screen/gws_MN_screen_processed_data/plate_"$k"/
	gcloud storage cp prob-maps.zip gs://lasagna-emil/genome-wide_MN_screen/gws_MN_screen_processed_data/plate_"$k"/
	cd ../
done
