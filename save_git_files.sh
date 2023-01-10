cp $(git status --porcelain=v1 | awk {'print $2'}) $1
#cp $(git status --porcelain=v1 | awk {'print $2'}) /workspace/playground/cpp/ln
