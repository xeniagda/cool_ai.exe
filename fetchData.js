var fs = require('fs');


var index = Math.floor(fs.list("data").length/2)-1;

//console.log(JSON.stringify(fs.list("data")));

getTest();

function getTest(){

	var token,lat,lng;

	function getData(callback){
		var page1 = require('webpage').create(),
		    server = 'https://geoguessr.com/api/v3/games/',
		    data = '{"map": "world", "type": "standard"}';
			var headers = {
			    "Content-Type": "application/json"
			}

		page1.open(server, 'post', data, headers, function (status) {
		    if (status !== 'success') {
		        console.log('Unable to post!');
		    } else {
		    	var str = (page1.content + "");
		        var data = JSON.parse(str.substring(str.indexOf("{"),str.lastIndexOf("}")+1));
		        //console.log(data.rounds[0].lat)
		        //console.log(data.rounds[0].lng)
		        //console.log(data.token)
		        lat = data.rounds[0].lat;
		        lng = data.rounds[0].lng;
		        token = data.token;
				callback();
		    }
		});
	}

	function getImage(callback){
		//console.log(token);
		var page = require('webpage').create();

		page.viewportSize = {
		  width: 1600,
		  height: 1000
		};

		page.captureContent = [/.*/];
		page.open("https://geoguessr.com/map/world", function(){
		    page.evaluate(function(token){
		        localStorage.setItem("last-game-timestamp-world",new Date().toString());
		        localStorage.setItem("last-game-world",token);
		    },token);
		    page.open("https://geoguessr.com/world/play", function(){
			    page.evaluate(function(){
		    		document.getElementsByClassName("button button--full-width modal__action-button")[0].click()
			    });
		        setTimeout(function(){  
		            page.render("data/"+index + ".png");
		            var data = JSON.stringify({
						lat:lat/90,
						lng:lng/180
					});
					fs.write('data/'+index, data,"w");
					callback();
		        },12000)
		    });    
		});
	}

	getData(function(){
		getImage(function(){
			console.log("Done index "+ index);
			index++;
			setTimeout(function(){
				getTest();
			},10);
		});
	});
}





/*
page.onResourceReceived = function(response) {

  //console.log('Response (#' + response.id + ', stage "' + response.stage + '"): ' + JSON.stringify(response));
  console.log(response.body)
};*/

/*var path = require('path')
var childProcess = require('child_process')
var phantomjs = require('phantomjs')
var binPath = phantomjs.path

var page = require('webpage').create();

page.open("https://sample.com/asdfasdf", function(){
    page.evaluate(function(){
        localStorage.setItem("something", "whatever");
    });

    page.open("https://sample.com", function(){
        setTimeout(function(){
            // Where you want to save it    
            page.render("screenshoot.png")  
            // You can access its content using jQuery
            var fbcomments = page.evaluate(function(){
                return $("body").contents().find(".content") 
            }) 
            phantom.exit();
        },1000)
    });    
});



/*const Pageres = require('pageres');

const pageres = new Pageres({delay: 2})
	.src('https://geoguessr.com/world/play', ['1024x768'],  {delay: 5})
	.dest(__dirname)
	.run()
	.then(() => console.log('done'));


/*var webshot = require('webshot');

var options = {
  screenSize: {
    width: 1000,
    height: 600
  },
	userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
	cookies:null,
	renderDelay:4000
};

webshot('https://geoguessr.com/world/play', 'test.png', options, function(err) {
  // screenshot now saved to google.png
});*/