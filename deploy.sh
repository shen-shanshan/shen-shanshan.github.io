#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# build the project
hugo -t blowfish

# cd public

git add .

msg="rebuilding site `date`"

if [ $# -eq 1 ]
  then msg="$1"
fi

git commit -m "$msg"

# push source to github

git push upstream master

# come back to blog root

# cd ..
