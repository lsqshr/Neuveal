path="$1"
find -name path

find $path -name '*.mat' | while read line; do
    echo "Processing file '$line'"
	luajit convert2t7.lua --path2file $line 
done
