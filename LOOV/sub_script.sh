# Submission script for the helloWorld program
#
# Comments starting with #OAR are used by the resource manager if using "oarsub -S"
#
#OAR -l /nodes=1
#
# The job is submitted to the default queue
#OAR -q default
# 
# Path to the binary to run
./start_imgnet.sh
