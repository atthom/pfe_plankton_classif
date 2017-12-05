# Submission script for the helloWorld program
#
# Comments starting with #OAR are used by the resource manager if using "oarsub -S"
#
#OAR -p gpu='YES' -l /nodes=1,walltime=30:00:00
#
# The job is submitted to the default queue
#OAR -q default
# 
# Path to the binary to run
./start_imgnet2.sh
