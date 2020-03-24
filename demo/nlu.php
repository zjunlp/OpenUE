<?php
$query = ""; 
$ip = "";
$port = "8676";
$channel = "200";
if($_REQUEST and isset( $_REQUEST["query1"])){
    $query1 = $_REQUEST['query1'];
}
if($_REQUEST and isset( $_REQUEST["ip"])){
    $ip= $_REQUEST['ip'];
}
if($_REQUEST and isset( $_REQUEST["channel"])){
    $channel= $_REQUEST['channel'];
}
if($_REQUEST and isset( $_REQUEST["model"])){
    $model= $_REQUEST['model'];
}
if($_REQUEST and isset( $_REQUEST["version"])){
    $version = $_REQUEST['version'];
}
?>
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>OpenUE Demo</title>
    <link rel="shortcut icon" href="" type="image/x-icon" />
    <!-- <link href="/static/css/bootstrap.css" rel="stylesheet"> -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.1.1/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0">
    <!-- <link rel="stylesheet" type="text/css" href="/static/css/flat-ui.min.css"> -->
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.3.1/dist/jquery.min.js"></script>
    <!-- <script src="/static/js/bootstrap.min.js"></script> -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.1.1/dist/js/bootstrap.min.js"></script>
    <meta name="google-site-verification" content="y4k1pfH9AoCgohmZAmIiAOYRR-pj6nAvmw9aijbmHtk" />
    <style>
        
        @font-face {font-family: "iconfont";
        src: url('//at.alicdn.com/t/font_792855_3rexpihinq3.eot?t=1534247389857'); /* IE9*/
        src: url('//at.alicdn.com/t/font_792855_3rexpihinq3.eot?t=1534247389857#iefix') format('embedded-opentype'), /* IE6-IE8 */
        url('data:application/x-font-woff;charset=utf-8;base64,d09GRgABAAAAAAQAAAsAAAAABnwAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAABHU1VCAAABCAAAADMAAABCsP6z7U9TLzIAAAE8AAAARAAAAFY8skguY21hcAAAAYAAAABLAAABcOdktl5nbHlmAAABzAAAAE0AAACIxEUujGhlYWQAAAIcAAAALQAAADYSUAmlaGhlYQAAAkwAAAAcAAAAJAfeA4NobXR4AAACaAAAAAgAAAAICAAAAGxvY2EAAAJwAAAABgAAAAYARAAAbWF4cAAAAngAAAAgAAAAIAEPADxuYW1lAAACmAAAAUUAAAJtPlT+fXBvc3QAAAPgAAAAHQAAAC5jcnN5eJxjYGRgYOBikGPQYWB0cfMJYeBgYGGAAJAMY05meiJQDMoDyrGAaQ4gZoOIAgCKIwNPAHicY2BkYWCcwMDKwMHUyXSGgYGhH0IzvmYwYuRgYGBiYGVmwAoC0lxTGBye2TyzYW7438AQw9zA0AAUZgTJAQDmhQxbeJxjYGBgZWBgYAZiHSBmYWBgDGFgZAABP6AoI1icmYELLM7CoARWwwISf2bz/z+MBPJZwCQDIxvDKOABkzJQHjisIJiBEQC7ogpZAHicY2BmAALm7UzvGPgZ5Bn0GRhURUXYlBWV1GwZTcyMFY3EBAnwmbeL8f+5wy8mxs+sAiKxsxs8QSwQwSjmxScmxgciGMXgogBe9BS0AAAAeJxjYGRgYADiBX/zC+L5bb4ycLMwgMD1GY/uItMsDEzvgBQHAxOIBwBY3QuHAAAAeJxjYGRgYG7438AQw8IAAkCSkQEVMAEARwgCawQAAAAEAAAAAAAAAABEAAAAAQAAAAIAMAADAAAAAAACAAAACgAKAAAA/wAAAAAAAHicZY9NTsMwEIVf+gekEqqoYIfkBWIBKP0Rq25YVGr3XXTfpk6bKokjx63UA3AejsAJOALcgDvwSCebNpbH37x5Y08A3OAHHo7fLfeRPVwyO3INF7gXrlN/EG6QX4SbaONVuEX9TdjHM6bCbXRheYPXuGL2hHdhDx18CNdwjU/hOvUv4Qb5W7iJO/wKt9Dx6sI+5l5XuI1HL/bHVi+cXqnlQcWhySKTOb+CmV7vkoWt0uqca1vEJlODoF9JU51pW91T7NdD5yIVWZOqCas6SYzKrdnq0AUb5/JRrxeJHoQm5Vhj/rbGAo5xBYUlDowxQhhkiMro6DtVZvSvsUPCXntWPc3ndFsU1P9zhQEC9M9cU7qy0nk6T4E9XxtSdXQrbsuelDSRXs1JErJCXta2VELqATZlV44RelzRiT8oZ0j/AAlabsgAAAB4nGNgYoAALgbsgImRiZGZgSUpsaiYgQEACckBwgAAAA==') format('woff'),
        url('//at.alicdn.com/t/font_792855_3rexpihinq3.ttf?t=1534247389857') format('truetype'), /* chrome, firefox, opera, Safari, Android, iOS 4.2+*/
        url('//at.alicdn.com/t/font_792855_3rexpihinq3.svg?t=1534247389857#iconfont') format('svg'); /* iOS 4.1- */
        }

        .iconfont {
        font-family:"iconfont" !important;
        font-size:16px;
        font-style:normal;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        }

        .icon-bars:before { content: "\e63c"; }

        html,body{
            margin:0;
        }
        body{
            font-family: PingFang SC, Lantinghei SC, Microsoft Yahei, Hiragino Sans GB, Microsoft Sans Serif, WenQuanYi Micro Hei, sans;
            background-color: #ededed;
            color:#060A0E;
            font-size: 14px;            
        }

        @media screen and (min-width: 1300px) {
            .container{
                max-width: 1300px;
            }
        }

        .site-header {
            background-color: #060a0e;
            font-size: 14px;
        }

        .site-header .nav-item{
            color: #c8c8c8;            
            padding: 0 15px;
        }


        .site-header .nav-link{
            cursor: pointer;
            text-decoration: none;
            position: relative;
            color: inherit;
            padding-left: 0!important;
            padding-right: 0!important;
        }


        .site-header .nav-item:hover,
        .site-header .nav-item.active{
            color: #fff;
        }

        .site-header .nav-link:hover::after,
        .site-header .active .nav-link::after{
            content: "";
            position: absolute;
            height: 2px;
            left: 0;
            right: 0;
            bottom: 3px;
            background-color: #fff;
        }

        
        @media screen and (max-width: 992px) {
            .site-header .nav-item.active,
            .site-header .nav-item:hover{
                background-color: #333;
            }
            .site-header .nav-link::after{
                display: none;
            }
        }

        .site-main{
            position: relative;
        }

        .footer{
            width: 100%;
            background: #060a0e;
            margin-top: 100px;
            color: #ededed;
            text-align: center;
            padding: 20px 0;
        }

        .footer p,
        .footer a{
            font-size: 12px;
            color: #fff;
            margin: 0;
            line-height: 20px;
        }

        .footer-logo{
            width: 100px;
        }

        .footer .beian:after {
            font-size: 0;
            /*内容为空*/
            content: " ";
            /*高度为父级元素的100%*/
            height: 100%;
        }
        .footer .beian{
            height: 30px;
            line-height: 30px;
        }

        a:hover {
            text-decoration: none;
        }
        .motto{
            border-top: 1px solid #ccc;
            margin-top: 40px;
            padding-top: 20px;
        }
        .motto-text{
            font-size: 14px;
            margin: 0; 
        }
    </style>

    <style>
        .entity-source{
            margin-right: 5px; 
            color: #470024;
        }
        .entity-info-img{
            width: 200px;
            margin-bottom: 15px;
        }
    </style>

    <script>
        window.__config__ = {
            fontFamily: 'PingFang SC, Lantinghei SC, Microsoft Yahei'
        }
    </script>
    <style type="text/css">
        .node {
          stroke: #fff;
          fill:#ddd;
          stroke-width: 1.5px;
        }

        .link {
          stroke: #999;
          stroke-opacity: .6;
        }

        marker {
            stroke: #999;
            fill:rgba(124,240,10,0);
        }

        .node-text {
          font: 11px sans-serif;
          fill:black;
        }

        .link-text {
          font: 9px sans-serif;
          fill:grey;
        }
        
        svg{
            border:1px solid black;
        }
    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
    <script src="https://d3js.org/d3.v3.min.js"></script>
    <script>
        function filterNodesById(nodes,id){
            return nodes.filter(function(n) { return n.id === id; });
        }
        
        function triplesToGraph(triples){
        
            svg.html("");
            //Graph
            var graph={nodes:[], links:[]};
            
            //Initial Graph from triples
            triples.forEach(function(triple){
                var subjId = triple.subject;
                var predId = triple.predicate;
                var objId = triple.object;
                
                var subjNode = filterNodesById(graph.nodes, subjId)[0];
                var objNode  = filterNodesById(graph.nodes, objId)[0];
                
                if(subjNode==null){
                    subjNode = {id:subjId, label:subjId, weight:1};
                    graph.nodes.push(subjNode);
                }
                
                if(objNode==null){
                    objNode = {id:objId, label:objId, weight:1};
                    graph.nodes.push(objNode);
                }
            
                
                graph.links.push({source:subjNode, target:objNode, predicate:predId, weight:1});
            });
            
            return graph;
        }
        
        
        function update(){
            // ==================== Add Marker ====================
            svg.append("svg:defs").selectAll("marker")
                .data(["end"])
              .enter().append("svg:marker")
                .attr("id", String)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 30)
                .attr("refY", -0.5)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
              .append("svg:polyline")
                .attr("points", "0,-5 10,0 0,5")
                ;
                
            // ==================== Add Links ====================
            var links = svg.selectAll(".link")
                                .data(graph.links)
                                .enter()
                                .append("line")
                                    .attr("marker-end", "url(#end)")
                                    .attr("class", "link")
                                    .attr("stroke-width",1)
                            ;//links
            
            // ==================== Add Link Names =====================
            var linkTexts = svg.selectAll(".link-text")
                        .data(graph.links)
                        .enter()
                        .append("text")
                            .attr("class", "link-text")
                            .text( function (d) { return d.predicate; })
                        ;

                //linkTexts.append("title")
                //      .text(function(d) { return d.predicate; });
                        
            // ==================== Add Link Names =====================
            var nodeTexts = svg.selectAll(".node-text")
                        .data(graph.nodes)
                        .enter()
                        .append("text")
                            .attr("class", "node-text")
                            .text( function (d) { return d.label; })
                        ;

                //nodeTexts.append("title")
                //      .text(function(d) { return d.label; });
            
            // ==================== Add Node =====================
            var nodes = svg.selectAll(".node")
                                .data(graph.nodes)
                                .enter()
                                .append("circle")
                                    .attr("class", "node")
                                    .attr("r",8)
                                    .call(force.drag)
                            ;//nodes
        
            // ==================== Force ====================
            force.on("tick", function() {
                nodes
                    .attr("cx", function(d){ return d.x; })
                    .attr("cy", function(d){ return d.y; })
                    ;
                
                links
                    .attr("x1",     function(d) { return d.source.x; })
                    .attr("y1",     function(d) { return d.source.y; })
                    .attr("x2",     function(d) { return d.target.x; })
                    .attr("y2",     function(d) { return d.target.y; })
                   ;
                   
                nodeTexts
                    .attr("x", function(d) { return d.x + 12 ; })
                    .attr("y", function(d) { return d.y + 3; })
                    ;
                    

                linkTexts
                    .attr("x", function(d) { return 4 + (d.source.x + d.target.x)/2  ; })
                    .attr("y", function(d) { return 4 + (d.source.y + d.target.y)/2 ; })
                    ;
            });
            
            // ==================== Run ====================
            force
              .nodes(graph.nodes)
              .links(graph.links)
              .charge(-500)
              .linkDistance(300)
              .start()
              ;
        }
        
        
    </script>
</head>

<body>
    <header class="site-header js-site-header">
        <nav class="navbar navbar-expand-lg navbar-default" style="height: 80px;">
                    <img src="logo.jpg" width="80" height="80"  style='position:absolute;left:3000;top:200'>
            <div class="container">
                <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon iconfont icon-bars" style="color: #fff;line-height: 24px;"></span>
                </button>
                
                <div class="collapse navbar-collapse justify-content-md-center" id="navbarSupportedContent">
                    <ul class="navbar-nav">
                            <li class="nav-item">
                                <a class="nav-link" href="index.php">Knowledge Extraction</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="med_triple.php">Triple Extraction(medical)</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="few_triple.php">Triple Extraction(few-shot)</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="event.php">Event Extraction</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="nlu.php">Natural Language Understanding</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="sentiment.php">Aspect-based Sentiment Analysis</a>
                            </li>
                    </ul>
                </div>
            </div>
        </nav>
    </header>


    <!--页面-->

        
<style>
    .infoball-wrap{
        margin: auto;
        height: 100%;
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        align-content: center;
    }
    .infoball{
        text-align: center;
        float: left;
        border-radius: 100%;
        background-color:#dddddd;
        height: 160px;
        width: 160px;
        font-size: 1em;
        font-weight:bold;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 10px solid;
        margin: 20px 0;
    }
    @media (max-width: 850px){
        .infoball{
            border-radius: 5px;
            height: auto;
            width: 100%;
            border-width: 2px;
            padding: 10px 0;
            margin: 15px 0;
        }
    }
    .infoball p{
        margin: 0;
    }
    .bignumber{
        font-size: 1.3em;
    }

    #svg-container svg{
        max-width: 400px;
  text-align: center;
    }
    #inner {
  text-align: center;
    }

</style>
<form action="med_triple.php" method="POST">
    <div id="interface" class="container py-5">
        <div class="row">
            <div class="col-12 d-flex justify-content-center">
                <div id="inputbox" class="input-group input-group-md" style="width: 860px; max-width: 500%;">
TBD
                </div>
            </div>
</form>
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

        </div>
    </div>


   <div class="js-search-guide container">
        <div class="card">
            <div class="inner" id="svg-body" align="center">

<?php
if(strlen($query1) > 0){
    $cmd = "python get_openue_med_triple.py \"$query1\" \"$model\"";
    $res = shell_exec($cmd);
    echo $res;
}
?>

        </div>
        </div>
    </div>


    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@3.5.5/d3.min.js"></script>
<script src="/static/js/countUp.js"></script>



    </div>
    <!--页脚-->
    <div class="footer">
        <div class="container">
            <div class="col-12 px-0">
                <!-- <img class="footer-logo" src="/static/img/actlogo.png"> -->
            </div>
        </div>
    </div>
    
    <script>
        
        $(".navbar-nav a[href='" + window.location.pathname + "']").parent().addClass("active");
        if(motto){
            $(".motto-text").text(motto);
        }else{
            $(".motto").hide();
        }
    </script>

    <script>
    // var _hmt = _hmt || [];
    // (function() {
    //   var hm = document.createElement("script");
    //   hm.src = "https://hm.baidu.com/hm.js?77060727ab59de1c09f5596807ac2249";
    //   var s = document.getElementsByTagName("script")[0]; 
    //   s.parentNode.insertBefore(hm, s);
    // })();
    </script>

</body>
</html>

