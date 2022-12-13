exp="baselines_1per"
device="cuda:1"

declare -a datasets=("Computers" "UWaveGestureLibraryAll" "Strawberry" "wafer" "yoga" "MiddlePhalanxOutlineCorrect" "FordA" "uWaveGestureLibrary_Z" "ScreenType" "SmallKitchenAppliances" "Two_Patterns" "DistalPhalanxOutlineCorrect" "HandOutlines" "FordB" "ProximalPhalanxOutlineAgeGroup" "ChlorineConcentration" "PhalangesOutlinesCorrect" "uWaveGestureLibrary_X" "RefrigerationDevices" "LargeKitchenAppliances" "uWaveGestureLibrary_Y" "ProximalPhalanxOutlineCorrect" "StarLightCurves" "ElectricDevices" "Earthquakes")
declare -a ssl_methods=("simclr" "cpc" "clsTran")

start=0
end=2

for dataset in "${datasets[@]}"
do
    for ssl_method in "${ssl_methods[@]}"
    do
      for i in $(eval echo {$start..$end})
      do
          python3 main.py --device $device --experiment_description $exp --fold_id $i --train_mode "ssl" \
          --data_percentage "100"  --sleep_model "cnn1d"  --ssl_method $ssl_method --dataset $dataset

          python3 main.py --device $device --experiment_description $exp --fold_id $i --train_mode "lc_1p" \
          --data_percentage "1"  --sleep_model "cnn1d" --ssl_method $ssl_method --dataset $dataset

          python3 main.py --device $device --experiment_description $exp --fold_id $i --train_mode "lc_5p" \
          --data_percentage "5"  --sleep_model "cnn1d" --ssl_method $ssl_method --dataset $dataset

      done
    done
done

