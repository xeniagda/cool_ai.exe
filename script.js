var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

var sz = 32;

var neural = new Neural(sz*sz*3,2,30,1,0.1,50,0,0);

var images = [];
var data = [];

load();

function load(){
	var index = 0;
	var loader = function(){
		if(index<=430)
			loadTest(index,loader);
		++index;
	};
	loader();
}

function loadTest(index,callback){
	var image = new Image();
	image.src = "compressed/" + index + ".png";
	image.onload = function(){
		ctx.drawImage(image,0,0);
		var pxl = ctx.getImageData(0, 0, sz, sz);

		var input = [];

		for (var i = 0; i < sz*sz*4; i++) {
			if(i%4!=3) input.push(pxl.data[i]/sz);
		}

		images.push(input);

		$.get("compressed/"+index,function(res){
			res = JSON.parse(res);
			data.push([(res.lat+1)/2,(res.lng+1)/2]);
			if(callback)
				callback();
		});
	}
}

var delay = 50;
var batchSize = 1000;
var index = 0;

var lastLoss = [];

function train(){
	var lossSum = 0;
	for (var i = 0; i < batchSize; i++) {
		lossSum += trainTest(index);
		index = (index+1)%400;
	}
	if(lastLoss.length>100)
		lastLoss.splice(0,1);
	lastLoss.push(avgScore(10,0,400));
	console.log(lossSum/batchSize,lastLoss.reduce((a,b)=>a+b,0)/lastLoss.length);
       
    addDataPoint(lastLoss.reduce((a,b)=>a+b,0)/lastLoss.length);
	setTimeout(train,delay);
}

function trainTest(index){
	return neural.next(images[index],data[index],1);
}

function avgScore(tests,a,b){
	var sum = 0;
	for (var i = 0; i < tests; i++) {
		sum+=getScore(a,b);
	}
	return sum/tests;
}

function getScore(a,b){
	var score = 0;
	for (var i = 0; i < 5; i++) {
		var testI = Math.floor(Math.random()*(b-a))+a;
		var guess = neural.test(images[testI]);
		var dist = coordDistance(data[testI][0]*180-90,data[testI][1]*360-180,guess[0]*180-90,guess[1]*360-180);
		score += 5000*Math.exp(-dist/2000);
		if(5000*Math.exp(-dist/2000)>3000){
			console.log("nice")
		}
	}
	return score;
}

var guessMarker, ansMarker;

function demonstrate(){
	var testI = Math.floor(Math.random()*30)+400;
	var image = new Image();
	image.src = "compressed/" + testI + ".png";
	image.onload = function(){
		ctx.drawImage(image,0,0);
		var pxl = ctx.getImageData(0, 0, sz, sz);

		var input = [];

		for (var i = 0; i < sz*sz*4; i++) {
			if(i%4!=3) input.push(pxl.data[i]/sz);
		}

		$.get("compressed/"+testI,function(res){
			res = JSON.parse(res);
			output = [(res.lat+1)/2,(res.lng+1)/2];
			var guess = neural.test(images[testI]);
			var dist = coordDistance(output[0]*180-90,output[1]*360-180,guess[0]*180-90,guess[1]*360-180);

			console.log(guess[0]*180-90 + "," + (guess[1]*360-180));
			console.log(output[0]*180-90 + "," + (output[1]*360-180));
			console.log(5000*Math.exp(-dist/2000));
			
			var pinColor = "FE7569";
	    	var pinImage = new google.maps.MarkerImage("http://chart.apis.google.com/chart?chst=d_map_pin_letter&chld=%E2%80%A2|" + pinColor,
	        new google.maps.Size(21, 34),
	        new google.maps.Point(0,0),
	        new google.maps.Point(10, 34));

			var from = {lat: guess[0]*180-90, lng: guess[1]*360-180};
			var to = {lat:output[0]*180-90, lng: output[1]*360-180};
		    if(guessMarker)
		    	guessMarker.setMap(null);
		    guessMarker = new google.maps.Marker({
		      position: from,
		      map: map
		    });
		    if(ansMarker)
		    	ansMarker.setMap(null);
		    ansMarker = new google.maps.Marker({
		      position: to,
		      map: map,
		      icon:pinImage
		    });

		});
	}
}

function coordDistance(lat1, lon1, lat2, lon2, unit) {
	var radlat1 = Math.PI * lat1/180
	var radlat2 = Math.PI * lat2/180
	var theta = lon1-lon2
	var radtheta = Math.PI * theta/180
	var dist = Math.sin(radlat1) * Math.sin(radlat2) + Math.cos(radlat1) * Math.cos(radlat2) * Math.cos(radtheta);
	dist = Math.acos(dist)
	dist = dist * 180/Math.PI
	dist = dist * 60 * 1.1515
	unit = "K";
	if (unit=="K") { dist = dist * 1.609344 }
	if (unit=="N") { dist = dist * 0.8684 }
	return dist
}


var container = document.getElementById('visualization');
var dataset = new vis.DataSet();

var options = {
    start: vis.moment().add(-50, 'seconds'),
    end: vis.moment().add(200, 'seconds'),
    dataAxis: {
        left: {
            range: {
                min: 0,
                max: 15000
            }
        }
    },
    drawPoints: false,
    shaded: {
        orientation: 'bottom'
    },
    interpolation: false
};
var graph2d = new vis.Graph2d(container, dataset, options);

function addDataPoint(y) {
    dataset.add({
        x: vis.moment(),
        y: y
    });
}

function setGraphMax(mx){
    var options = {
        start: graph2d.getDataRange().min,
        end: graph2d.getDataRange().max,
        dataAxis: {
            left: {
                range: {
                    min: 0,
                    max: mx
                }
            }
        },
        drawPoints: false,
        shaded: {
            orientation: 'bottom'
        },
        interpolation: false
    };
    graph2d.setOptions(options);
}

var map;

function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
		      zoom: 1,
		      center: {lat:0,lng:0}
		    });
 }