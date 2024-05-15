#!/bin/bash

timestamp() {
  date +"%Y-%m-%d%H%M%S"
}

error_exit()
{
#   ----------------------------------------------------------------
#   Function for exit due to fatal program error
#       Accepts 1 argument:
#           string containing descriptive error message
#   Source: http://linuxcommand.org/lc3_wss0140.php
#   ----------------------------------------------------------------
    echo "$(timestamp) ERROR ${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    echo "$(timestamp) ERROR ${PROGNAME}: Exiting Early."
    exit 1
}

error_check()
{
#   ----------------------------------------------------------------
#   This function simply checks a passed return code and if it is
#   non-zero it returns 1 (error) to the calling script.  This was
#   really only created because I needed a method to store STDERR in
#   other scripts and also check $? but also leave open the ability
#   to add in other stuff, too.
#
#   Accepts 1 arguments:
#       return code from prior command, usually $?
#  ----------------------------------------------------------------
    TO_CHECK=${1:-0}

    if [ "$TO_CHECK" != '0' ]; then
        return 1
    fi

}

get_mode()
{
  if [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "True" ]]; then
    echo "predcls"
  elif [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    echo "sgcls"
  elif [[ ${USE_GT_BOX} == "False" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    echo "sgdet"
  else
    error_exit "Illegal USE_GT_BOX=${USE_GT_BOX} and USE_GT_OBJECT_LABEL=${USE_GT_OBJECT_LABEL} provided."
  fi
}