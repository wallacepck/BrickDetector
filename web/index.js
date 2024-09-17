eel.setup()

$(document).ready(function(){
  $('#elapsed').tooltip();

  $('#demoPanel').on('hide.bs.collapse', function () {
    eel.hookDemo("output");
  });
  
  $('#demoPanel').on('show.bs.collapse', function () {
    eel.hookDemo(lastHook);
  });
});

eel.expose(updateImageSrc)
function updateImageSrc(val, id) {
  let elem = document.getElementById(id);
  if (val == "") 
  {
    elem.src = "";
  }
  else
  {
    elem.src = "data:image/jpeg;base64," + val;
  }
  
}

eel.expose(setTransparency)
function setTransparency(val, id) {
  document.getElementById(id).style = `opacity: ${val};`;
}

eel.expose(addDetCategory)
function addDetCategory(color, name, id, b, a) {
  x = a*1.5
  y = b*1.5

  let section = document.getElementById("categoryList");

  let item = document.createElement("LI");
  item.classList.add("list-group-item");
  item.classList.add("custom-bg-dark-fg");
  item.classList.add("no-select");
  item.classList.add("p-2");

  let transparency = document.createElement("DIV");
  transparency.classList.add("d-flex");
  transparency.classList.add("justify-content-between");
  transparency.style = "opacity: 0.1;";
  transparency.id = id + "_"

  let div = document.createElement("DIV");
  div.classList.add("my-auto") 
  div.style = `background-color: ${color}; width: ${x}vh; height: ${y}vh;`;
  div.setAttribute("data-toggle", "tooltip")
  div.setAttribute("title", `${name}`)

  let para = document.createElement("H6");
  para.classList.add("text-white");
  para.style = `line-height: ${y}vh; text-align: center; text-shadow: 0px 0px 6px black;`;
  para.innerHTML = b + "x" + a;

  let header = document.createElement("H3");
  header.classList.add("my-auto");

  let badge = document.createElement("SPAN");
  badge.classList.add("badge");
  badge.classList.add("badge-dark");
  badge.innerHTML = "0";
  badge.id = id;

  header.appendChild(badge);

  div.appendChild(para);

  transparency.appendChild(div);
  transparency.appendChild(header);

  item.append(transparency);
  section.appendChild(item);

  // <li class="list-group-item d-flex justify-content-between custom-bg-dark-fg no-select p-2">
  //     <div class="my-auto" style="background-color: coral; width: 12vh; height: 8vh;" data-toggle="tooltip" data-placement="top" title="">
  //         <h2 class="text-white" style="line-height: 8vh;text-align: center; text-shadow: 0px 0px 6px black;">2x4</h2>
  //     </div>

  //     <h1 class="my-auto"> 
  //         <span class="badge badge-dark">4</span>
  //     </h1>
  // </li>
}

eel.expose(addTabToDemoPanel)
function addTabToDemoPanel(internalName, metadata, selector=null, inline=true) {
  let text = metadata.text
  let externalName = metadata.name
  if ('selector' in metadata) selector = metadata.selector
  if ('inline' in metadata) inline = metadata.inline

  let section = document.getElementById("demoTabList");

  let navItem = document.createElement("LI");
  navItem.classList.add("nav-item");

  let anchor = document.createElement("A");
  anchor.classList.add("nav-link");
  anchor.classList.add("text-light");
  anchor.setAttribute("data-toggle", "tab");
  anchor.href = `#${internalName}`;
  anchor.innerHTML = externalName;
  anchor.style.height = "100%";
  
  section.appendChild(navItem);
  navItem.appendChild(anchor);


  let tabContent = document.getElementById("demoContentPanel")

  let tabPane = document.createElement("DIV");
  tabPane.classList.add("tab-pane")
  tabPane.classList.add("container")
  tabPane.classList.add("fade")
  tabPane.id = internalName

  let para = document.createElement("P");
  para.className = "text-white";
  para.innerHTML = text;
  para.style = "white-space: pre-line;"

  tabPane.appendChild(para);

  if (selector) {
    let firstCheck = true;
     if (Object.keys(selector).length > 1) {
      anchor.setAttribute("onclick", `updateHook(getCheckedValue("${internalName}"))`);

      for (var key in selector) {
        let div = document.createElement("DIV");
        div.classList.add("custom-control");
        div.classList.add("custom-radio");
        if (inline) div.classList.add("custom-control-inline");
    
        let radio = document.createElement("INPUT");
        radio.setAttribute("type", "radio");
        radio.setAttribute("onclick", `updateHook("${key}")`);
        if (firstCheck) {
          radio.setAttribute("checked", "");
          firstCheck = false;
        }
        radio.classList.add("custom-control-input");
        radio.id = internalName + key;
        radio.name = internalName;
        radio.value = key;

        let label = document.createElement("LABEL");
        label.classList.add("custom-control-label");
        label.classList.add("text-white");
        label.setAttribute("for", internalName + key);
        label.innerHTML = selector[key];
    
        div.appendChild(radio);
        div.appendChild(label);
        tabPane.appendChild(div);
      }
    } else if (Object.keys(selector).length == 1) {
      anchor.setAttribute("onclick", `updateHook("${Object.keys(selector)[0]}")`);
    }
  }

  tabContent.appendChild(tabPane);
}

var lastHook = null;
function updateHook(hook) {
  lastHook = hook;
  eel.hookDemo(hook);
}

function getCheckedValue(name) {
  let input = document.querySelector(`input[name=${name}]:checked`);
  if (input) return input.value;
  return null;
}

eel.expose(addVideoToList)
function addVideoToList(group, title, thumb) {
  let section = document.getElementById(group);
  let item = document.createElement("LI");
  item.classList.add("list-group-item");
  item.classList.add("custom-text-ash");
  item.classList.add("custom-bg-dark-fg");

  let text = document.createTextNode(title);

  let img = document.createElement("IMG");
  img.src = "data:image/jpeg;base64," + thumb;
  img.className = "scale-to-viewport";
  img.style = "cursor: pointer;";
  img.setAttribute("onclick", `eel.videoSelect("${title}")`);

  item.appendChild(text);
  item.appendChild(img);

  section.appendChild(item);
}

var slow = false
eel.expose(redTime)
function redTime(val) {
  let text = document.getElementById("elapsed");
  if (val && !slow) {
    slow = true
    text.classList.add("text-danger");
    text.classList.remove("text-white");
    text.setAttribute("title", "Image processing is slower than 10 FPS.");
    text.setAttribute("data-original-title", "Image processing is slower than 10 FPS.");
  } else if (!val && slow) {
    slow = false
    text.classList.add("text-white");
    text.classList.remove("text-danger");
    text.setAttribute("title", "");
    text.setAttribute("data-original-title", "");
  }
}

eel.expose(loadingBarUpdate)
function loadingBarUpdate(val) {
  if (val == null) {
    document.getElementById("loadingBar").classList.add("invisible")
  } else {
    document.getElementById("loadingText").innerHTML = `Loading : ${val}`
  }
}

eel.expose(seekBarMax)
function seekBarMax(val) {
  document.getElementById("seek-bar").max=val;
}

eel.expose(seekBarValue)
function seekBarValue(val) {
  if (val == null) return document.getElementById("seek-bar").value;
  document.getElementById("seek-bar").value=val;
}

eel.expose(showPauseButton)
function showPauseButton(playing) {
  document.getElementById("playButton").className = playing ? "fas fa-pause" : "fas fa-play";
}

eel.expose(updateVideoSrc)
function updateVideoSrc(val, id) {
  document.getElementById(id).src = val
}
eel.expose(updateTextSrc)
function updateTextSrc(val,id) {
  document.getElementById(id).innerHTML = val;
}
eel.expose(updateImageSrc)
function updateImageSrc(val, id) {
  let elem = document.getElementById(id);
  elem.src = "data:image/jpeg;base64," + val;
}

function py_video() {
   eel.video_feed()()
}

var seeking = false;
var seekClick = false;
eel.expose(seek)
function seek(state) {
  seeking = state;
  if (state) seekClick = true;
}
eel.expose(acknowledgeSeekClick)
function acknowledgeSeekClick() {
  seekClick = false
}

eel.expose(isGUISeeking)
function isGUISeeking() {
  return seeking || seekClick;
}

$(window).keypress(function (e) {
  if (e.key === ' ') {
    eel.toggle_video_feed()
  }
});

eel.expose(get_Value)
function get_Value(id) {
  selectedVal= document.getElementById(id).innerHTML
  return selectedVal;
}
