imname=$1

DIRR=/Users/evcu/GitHub/evcu.github.io/assets/nyc365blog
CURR="$(pwd)"
REL=assets/nyc365blog
TODAY="$(date +"%m-%d-2017")"
python $DIRR/newDay.py || exit 1
echo $TODAY
echo $CURR
extension="${1##*.}"
extlower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
sips -Z 640 $1
mv $1 $DIRR/images/$TODAY.$extlower

cd /Users/evcu/GitHub/evcu.github.io/
echo $REL/images/$TODAY.jpg
git add .
git commit -am 'Todays post to nyc365blog'
git status
git push
cd $CURR
