var ImageResize = require('node-image-resize');
var fs = require('fs');
var spawn = require('child_process').spawn;
var im = require('imagemagick');
var process = require("process")

var amount = process.argv[2] | 0;

var file_idx = 0;

function process_file(file) {
    if (file.endsWith(".png")) {

        while (fs.existsSync("data/tmp/" + file_idx))
            file_idx += 1;

        let file1 = 'data/load/' + file;
        let file2 = 'data/tmp/' + file_idx + ".png";

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

                fs.unlink(file1,   err => { if (err) console.log(err) });
                fs.unlink(coords1, err => { if (err) console.log(err) });
            });
        });

        // Copy coords
        var coords1 = 'data/load/' + file.replace(/\.png/, "");
        var coords2 = 'data/tmp/' + file_idx;

        console.log(coords1, "->", coords2);
        fs.createReadStream(coords1).pipe(fs.createWriteStream(coords2));
        return true;
    }
    return false;
}

fs.readdir("data/load", (err, files) => {
    if (err) console.log("Error:", err);

    if (amount == 0) {
        files.forEach(process_file);
    } else {
        var im = 0;
        for (var i = 0; im < amount && i < files.length; i++) {
            if (process_file(files[i])) { im++; }
        }
    }

})

