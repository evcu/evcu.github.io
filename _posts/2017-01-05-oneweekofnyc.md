---
layout: single
title: "One Week of NYC"
tags: [data, visualization, d3, data visualization]
category: datavis
excerpt: "Time spent on various activites throughtout a week visualized with a hierarchical pie chart"
---

I've recorded what I did in a week during my first semester at NYU. I've gathered various activities around 5 main category. 

- **Survival:** Sleep, Eat, Commute, 
- **Academical:** Lecture, Homework, Project
- **Self Development:** Individual projects, Sports, learning
- **Social:** Time spent with friends, skype talks
- **Fun:** Surfing, NFL, Shopping

I've used Python to pre-process the text data I had and save it as json file. 

<p>I've used the Andreas Dewes's <a href="https://bl.ocks.org/adewes/4710330" target="_blank">Hierarchical Pie Chart</a> for visualization.</p> 

Click slices to zoom-in, click center to zoom-out

<style>

#code_hierarchy
{
    position:relative;
    width:600px;
    margin:0 auto;
}

#code_hierarchy_legend
{
    height:100px;
    font-size:1.4em;
    text-align:center;
}
</style>

<script src="https://d3js.org/d3.v3.js"></script>
<script lang="text/javascript">

function init_code_hierarchy_plot(element_id,data,count_function,color_function,title_function,legend_function)
{
    var plot = document.getElementById(element_id);

    while (plot.hasChildNodes())
    {
        plot.removeChild(plot.firstChild);
    }

    var width = plot.offsetWidth;
    var height = width;
    var x_margin = 40;
    var y_margin = 40;
    
    var max_depth=3;
    
    var data_slices = [];
    var max_level = 4;

    var svg = d3.select("#"+element_id).append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + width / 2 + "," + height * .52 + ")");
          
    function process_data(data,level,start_deg,stop_deg)
    {
        var name = data[0];
        var total = count_function(data);
        var children = data[2];
        var current_deg = start_deg;
        if (level > max_level)
        {
            return;
        }
        if (start_deg == stop_deg)
        {
            return;
        }
        data_slices.push([start_deg,stop_deg,name,level,data[1]]);
        for (var key in children)
        {
            child = children[key];
            var inc_deg = (stop_deg-start_deg)/total*count_function(child);
            var child_start_deg = current_deg;
            current_deg+=inc_deg;
            var child_stop_deg = current_deg;
            var span_deg = child_stop_deg-child_start_deg;
            process_data(child,level+1,child_start_deg,child_stop_deg);
        }
    }
    
    process_data(data,0,0,360./180.0*Math.PI);

    var ref = data_slices[0];
    var next_ref = ref;
    var last_refs = [];

    var thickness = width/2.0/(max_level+2)*1.1;
        
    var arc = d3.svg.arc()
    .startAngle(function(d) { if(d[3]==0){return d[0];}return d[0]+0.01; })
    .endAngle(function(d) { if(d[3]==0){return d[1];}return d[1]-0.01; })
    .innerRadius(function(d) { return 1.1*d[3]*thickness; })
    .outerRadius(function(d) { return (1.1*d[3]+1)*thickness; });    

    var slices = svg.selectAll(".form")
        .data(function(d) { return data_slices; })
        .enter()
        .append("g");
        slices.append("path")
        .attr("d", arc)
        .attr("id",function(d,i){return element_id+i;})
        .style("fill", function(d) { return color_function(d);})
        .attr("class","form");
    slices.on("click",animate);

    if (title_function != undefined)
    {
        slices.append("svg:title")
              .text(title_function);        
    }
    if (legend_function != undefined)
    {
        slices.on("mouseover",update_legend)
              .on("mouseout",remove_legend);
        var legend = d3.select("#"+element_id+"_legend")
            
        function update_legend(d)
        {
            legend.html(legend_function(d));
            legend.transition().duration(200).style("opacity","1");
        }
        
        function remove_legend(d)
        {
            legend.transition().duration(1000).style("opacity","0");
        }
    }
    function get_start_angle(d,ref)
    {
        if (ref)
        {
            var ref_span = ref[1]-ref[0];
            return (d[0]-ref[0])/ref_span*Math.PI*2.0
        }
        else
        {
            return d[0];
        }
    }
    
    function get_stop_angle(d,ref)
    {
        if (ref)
        {
            var ref_span = ref[1]-ref[0];
            return (d[1]-ref[0])/ref_span*Math.PI*2.0
        }
        else
        {
            return d[0];
        }
    }
    
    function get_level(d,ref)
    {
        if (ref)
        {
            return d[3]-ref[3];
        }
        else
        {
            return d[3];
        }
    }
    
    function rebaseTween(new_ref)
    {
        return function(d)
        {
            var level = d3.interpolate(get_level(d,ref),get_level(d,new_ref));
            var start_deg = d3.interpolate(get_start_angle(d,ref),get_start_angle(d,new_ref));
            var stop_deg = d3.interpolate(get_stop_angle(d,ref),get_stop_angle(d,new_ref));
            var opacity = d3.interpolate(100,0);
            return function(t)
            {
                return arc([start_deg(t),stop_deg(t),d[2],level(t)]);
            }
        }
    }
    
    var animating = false;
    
    function animate(d) {
        if (animating)
        {
            return;
        }
        animating = true;
        var revert = false;
        var new_ref;
        if (d == ref && last_refs.length > 0)
        {
            revert = true;
            last_ref = last_refs.pop();
        }
        if (revert)
        {
            d = last_ref;
            new_ref = ref;
            svg.selectAll(".form")
            .filter(
                function (b)
                {
                    if (b[0] >= last_ref[0] && b[1] <= last_ref[1]  && b[3] >= last_ref[3])
                    {
                        return true;
                    }
                    return false;
                }
            )
            .transition().duration(1000).style("opacity","1").attr("pointer-events","all");
        }
        else
        {
            new_ref = d;
            svg.selectAll(".form")
            .filter(
                function (b)
                {
                    if (b[0] < d[0] || b[1] > d[1] || b[3] < d[3])
                    {
                        return true;
                    }
                    return false;
                }
            )
            .transition().duration(1000).style("opacity","0").attr("pointer-events","none");
        }
        svg.selectAll(".form")
        .filter(
            function (b)
            {
                if (b[0] >= new_ref[0] && b[1] <= new_ref[1] && b[3] >= new_ref[3])
                {
                    return true;
                }
                return false;
            }
        )
        .transition().duration(1000).attrTween("d",rebaseTween(d));
        setTimeout(function(){
            animating = false;
            if (! revert)
            {
                last_refs.push(ref);
                ref = d;
            }
            else
            {
                ref = d;
            }
            },1000);
    };    

}

function init_plots()
{
    
    function count_function(d)
    {
        return d[1];
    }
    
    function label_function(d)
    {
        return d[2]+": "+d[4]+" minutes.";
    }
    
    function legend_function(d)
    {
        return "<h2>"+d[2]+"&nbsp;</h2><p>"+d[4]+" minutes</p>"
    }
    
    var color = d3.scale.category20c();

    function color_function(d)
    {
        return color(d[2]);
    }
    d3.select(self.frameElement).style("height", "800px");
    init_code_hierarchy_plot("code_hierarchy",code_hierarchy_data,count_function,color_function,label_function,legend_function);
}

window.onload = init_plots;
window.onresize = init_plots;

</script>
<script type="text/javascript" src="/assets/data/oneweek.js"></script>
<div id="code_hierarchy_legend">&nbsp;</div>
<div id="code_hierarchy">&nbsp;</div>

Here is an excerpt from the raw file I've logged through the week

```
day 1 tuesday
wake 8.2
bfast 8.2 9
hwA 9 9.2
comm 9.2 9.3
web 9.3 10
hwM 10 12
...
comm 19.4 20.1
dinner 20.2 21
web 21 22
proj 22 22.2
hwM 22.2 23.3
web 23.3 24
sleep 24.1
```

I defined the hierarchy of activites and their descriptions with another text file.

```
-,survival,Survival Activities
bfast,Breakfast
sleep,Sleeping
lunch,Lunch
comm,Commute
dinner,Dinner
-,social,Social Activites
skypeFa,Skype with Family
friends,Meeting with Friends
skypeF,Skype with Friends
skypeJ,Skype with Julia'
...
```

I've wrote a python script to pre-process the raw text data to a json file according to the hiearchy above. Final json file looks like the following.

```
["One Week of NYC", 
9630, 
{"fun": ["Fun Activities", 
    610, 
    {"shop": ["Shopping", 
        60,
        {}],
    "web": ["Web Surfing", 
        380, 
        {}],
     "nfl": ["Watching NFL", 
        170, 
        {}] 
    }], 
"academic": ["Academical Activities", 
...
```