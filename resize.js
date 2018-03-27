var ImageResize = require('node-image-resize'),
    fs = require('fs');
 
var im = require('imagemagick');

for (var i = 400; i < 431; i++) {
    (function(){
        var file1 = 'data/'+i + ".png";
        var file2 = 'compressed/'+i + ".png";
        im.resize({
          srcPath: file1,
          dstPath: file2,
          width: 32,
          height: 32
        }, function(err, stdout, stderr){
          //im.convert([file2, '-crop', '32x32+64+0',file2],function(err, stdout, stderr){
              // foo 
            //});
            im.convert([file2, '-crop', '32x32','-gravity', 'center', '-background', 'black','-extent','32x32',file2],function(err, stdout, stderr){
              // foo 
            });
        });
    })();
}