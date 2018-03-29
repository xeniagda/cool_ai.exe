var ImageResize = require('node-image-resize'),
    fs = require('fs');
var spawn = require('child_process').spawn;

var im = require('imagemagick');


fs.readdir("data", (err, files) => {
    if (err) console.log("Error:", err);

    var file_idx = 0;

    files.forEach(file => {
        if (file.endsWith(".png")) {

            while (fs.existsSync("compressed/" + file_idx))
                file_idx += 1;

            let file1 = 'data/' + file;
            let file2 = 'compressed/' + file_idx + ".png";

            console.log(file1, "->", file2);

            // Resize image
            im.resize({
              srcPath: file1,
              dstPath: file2,
              width: 32,
              height: 32
            }, function(err, stdout, stderr){
                if (err) console.log("Error:", err);
                // im.convert([file2, '-crop', '32x32+64+0',file2],function(err, stdout, stderr){
                // });

                im.convert([file2, '-crop', '32x32','-gravity', 'center', '-background', 'black','-extent','32x32',file2],function(err, stdout, stderr){
                    if (err)
                        console.log(err);
                });
            });

            // Copy coords
            var coords1 = 'data/' + file.replace(/\.png/, "");
            var coords2 = 'compressed/' + file_idx;

            console.log(coords1, "->", coords2);
            fs.createReadStream(coords1).pipe(fs.createWriteStream(coords2));

        }
    })

})
