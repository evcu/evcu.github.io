---
layout: single
title: "365 Days in New York City"
permalink: /nyc365blog/
---

This is the blog **pretty compact**. I am planning to make it better during _the year_. But I should admit it is already _damm good_. **Click** any available square to see the data!

<link rel="stylesheet" href="/assets/nyc365blog/main.css">
<script src="https://d3js.org/d3.v3.min.js" charset="utf-8"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/3.10.1/lodash.min.js" charset="utf-8"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/moment.js/2.10.6/moment.min.js"></script>
  <article>
    <section id="heatmap" width="100%" >
<span id="info">info</span>
    </section>
    <table id="mytable" width="100%">
    <tbody>
    <tr>
    <td width="33%" ><div id="mytablediv" >
    <span id="day_hl">Highlights</span></div></td>
    <td width="33%"><div id="mytablediv">
    <span id="day_ll">Lowlights</span></div></td>
    <td width="34%"><div id="mytablediv" >
    <span id="day_ot">Other</span></div></td>
    </tr>
    <tr>
    <td colspan="2">
    <div id="mytablediv">
     <span id="day_rt">Random thoughts</span></div></td>
    <td width="34%"><div id="mytablediv" >
    <img src="/assets/images/blogimg/nyc365.gif" id="day_pic"></div></td>
    </tr>
    </tbody>
    </table>
  </article>
       

  <script> d3.eesur = {}; //namespace  </script>
  <script src="/assets/nyc365blog/d3_code_heatmap_cal.js"></script>
  <script>
  // *****************************************
  // render chart
  // *****************************************
  (function() {
      'use strict'; 
      var nestedData;
      var nestedText;
      var parseDate = d3.time.format('%Y-%m-%d').parse;
      // create chart
      var heatChart = d3.eesur.heatmap()
          .colourRangeStart('#e48c5c')
          .colourRangeEnd('#b1bf52')
          .height(130)
          .width("100%")
          .startYear('2017')
          .endYear('2018')
          .on('_hover', function (d, i) {
              var f = d3.time.format('%B %d, %Y');
              var myf = d3.time.format('%m-%d-%Y');
              d3.select('#info')
                  .text(function () {
                      var youare;
                      if (nestedData[d]>0){
                          youare="happy";
                        } else if (nestedData[d]<0) {
                          youare="sad";
                          } else{
                          youare="neutral";
                        }
                      return 'date: ' + f(d) + ' and you are ' + youare;
                  });
             d3.select("#day_pic")
              .attr("src","/assets/nyc365blog/images/"+myf(d)+".jpg");
          })
          .on('_click', function (d, i) {
              d3.select('#day_ot')
                  .text(function () {
                      return nestedText[d][0].other;
                  });
              d3.select('#day_rt')
                  .text(function () {
                      return nestedText[d][0].text;
                  });
              d3.select('#day_hl')
                  .text(function () {
                      return nestedText[d][0].high;
                  });
              d3.select('#day_ll')
                  .text(function () {
                      return nestedText[d][0].low;
                  });
          });
      // apply after nesting data
      d3.json('/assets/nyc365blog/data.json', function(error, data) {
          if (error) return console.warn(error);    
          nestedData = d3.nest()
              .key(function (d) { return parseDate(d.date.split(' ')[0]); })
              .rollup(function (n) { 
                  return d3.sum(n, function (d) { 
                      return d.mood; // key
                  }); 
              })
              .map(data);
          nestedText = d3.nest()
              .key(function (d) { return parseDate(d.date.split(' ')[0]); })
              .map(data);
          // render chart
          d3.select('#heatmap')
              .datum(nestedData)
              .call(heatChart);
      });
  }());
  d3.select(self.frameElement).style('height', '900px');

  </script>