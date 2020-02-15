#!/usr/bin/env bash

# example of usage 
## suppress warnings on call
start=`date +%s`
#echo pipeline init
#while true;do echo -n '>';sleep 1;done &



#kill $!; trap 'kill $!' SIGTERM
#echo
#echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime