# Compile a beamer-style PDF:
```
pandoc slides.md -t beamer -o c3se-intro.pdf -fmarkdown-implicit_figures
```

# Compile a reaveal-js HTML presentation:
```
wget https://github.com/hakimel/reveal.js/archive/master.tar.gz
tar -xzvf master.tar.gz
mv reveal.js-master reveal.js

pandoc slides.md -t revealjs -s -o c3se-intro.html -V revealjs-url=file://$HOME/c3se-documentation-git/docs/documentation/intro/reveal.js/ -V theme=c3se -fmarkdown-implicit_figures --highlight-style kate
```
