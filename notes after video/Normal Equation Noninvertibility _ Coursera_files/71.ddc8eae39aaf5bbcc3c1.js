(window.webpackJsonp=window.webpackJsonp||[]).push([[71],{"+o2K":function(module,exports,e){},"/0z3":function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("sbe7"),a=t.n(s),o=t("w/1P"),i=t.n(o),c=t("y1LI"),u=t("9A5E"),l=t("GP6s"),m=t.n(l),p=t("9F1q"),d=t.n(p),f=function(e){function DiscussionsGuidelines(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsGuidelines,e),DiscussionsGuidelines.prototype.render=function render(){var e=i()("rc-DiscussionsGuidelines","card-one-clicker","cozy","horizontal-box","align-items-spacebetween","align-items-vertical-center");return a.a.createElement(u.a,{className:"nostyle",trackingName:"DiscussionsGuidelines",href:"https://learner.coursera.help/hc/articles/208280036",target:"_blank"},a.a.createElement("div",{className:e},a.a.createElement("span",{className:"body-1-text"},m()("Forum guidelines")),a.a.createElement(c.a,{name:"chevron-right-thin",size:"lg"})))},DiscussionsGuidelines}(a.a.Component);e.a=f},"0Xjo":function(module,exports,e){var t=e("8Yfy"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"1Ovw":function(module,e,t){"use strict";t.d(e,"a",function(){return F});var n=t("pVnL"),r=t.n(n),s=t("MVZn"),a=t.n(s),o=t("VbXa"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("17x9"),m=t.n(l),p=t("VYij"),d=t.n(p),f=t("ZzvC"),h=t("GnyC"),b=t("b+bd"),y=t("oQ9N"),v=t("VXRf"),g=t("ngyh"),E=t.n(g),x=t("UU2Q"),C=function(e){function DiscussionsCurrentWeekLoader(){return e.apply(this,arguments)||this}var t;return i()(DiscussionsCurrentWeekLoader,e),DiscussionsCurrentWeekLoader.prototype.render=function render(){var e=this.props,t=e.weekNumber,n=e.forum,r=e.forumStatistic;if(!t||!n)return null;return u.a.createElement(f.a,{weekNumber:t,isLoading:!1,title:n.title,description:n.description,mostRecentThread:n.mostRecentThread,lastAnsweredAt:r&&r.lastAnsweredAt,forumQuestionCount:r&&r.forumQuestionCount})},DiscussionsCurrentWeekLoader}(u.a.Component),N=d.a.compose(Object(y.a)({fields:["link","title","forumType","description","lastAnsweredAt","forumQuestionCount"]}),Object(b.a)(["CourseStore","CourseScheduleStore","ProgressStore"],function(e,t){var n=e.CourseStore,r=e.CourseScheduleStore,s=e.ProgressStore,a;return{weekNumber:t.weekNumber||Object(h.a)(n,r,s)}}),Object(v.a)(function(e){var t=e.courseForums.find(function(t){return t.forumType.definition.weekNumber===e.weekNumber});return a()({},e,{forum:t,forumStatistic:t&&e.courseForumStatistics&&e.courseForumStatistics.find(function(e){return e.courseForumId===t.id})})}))(C),F=function(e){function FluxibleDiscussionsAppLoader(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).fluxibleContext=null,t}i()(FluxibleDiscussionsAppLoader,e);var t=FluxibleDiscussionsAppLoader.prototype;return t.componentWillMount=function componentWillMount(){var e=this.context.fluxibleContext;this.fluxibleContext=Object(x.a)(e)},t.render=function render(){return u.a.createElement(E.a,{context:this.fluxibleContext.getComponentContext()},u.a.createElement(N,r()({},this.props,{isCourseWeekPage:!0})))},FluxibleDiscussionsAppLoader}(u.a.Component);F.contextTypes={fluxibleContext:m.a.object}},"2l4Y":function(module,exports,e){},"3/q2":function(module,exports,e){var t=e("5gQg"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"3CU7":function(module,exports,e){var t=e("Xf+T"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"5fK1":function(module,exports,e){var t=e("G92m"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"5gQg":function(module,exports,e){},7450:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("Akn8"),m=t("wVBo"),p=t("ekry"),d=t("BZ+2"),f=t("Fi0Z"),h=t("oQ9N"),b=t("SaLm"),y=t("hN99"),v=t("GP6s"),g=t.n(v),E=t("0Xjo"),x=t.n(E),C=function(e){function DiscussionsWeekForums(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsWeekForums,e),DiscussionsWeekForums.prototype.render=function render(){var e=this,t=this.props.courseForums.filter(function(e){return e.forumType.typeName===b.j.weekForumType}).map(function(t){var n=e.props.courseForumStatistics.find(function(e){return e.courseForumId===t.id});return Object.assign({},t,{discussionsLink:l.a.join(Object(y.e)(e.props.courseSlug),t.link),lastAnsweredAt:n&&n.lastAnsweredAt,forumQuestionCount:n&&n.forumQuestionCount})});return u.a.createElement(m.a,{className:"rc-DiscussionsWeekForums",title:g()("Week Forums")},u.a.createElement("ul",null,t.map(function(e){return u.a.createElement("li",{key:e.id},u.a.createElement(p.a,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))})))},DiscussionsWeekForums}(u.a.Component);C.propTypes={courseForums:i.a.arrayOf(i.a.instanceOf(d.a)),courseForumStatistics:i.a.arrayOf(i.a.instanceOf(f.a)),courseSlug:i.a.string},e.a=Object(s.compose)(Object(h.a)({fields:["link","title","description","lastAnsweredAt","forumQuestionCount"]}))(C)},"8Yfy":function(module,exports,e){},"8wD4":function(module,exports,e){var t=e("x9/O"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"9F1q":function(module,exports,e){var t=e("o3Oq"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},COA5:function(module,exports,e){},"Dr+H":function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("w/1P"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("Qnnn"),m=t("w1w6"),p=t("/0z3"),d=function(e){function DiscussionsSideColumn(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsSideColumn,e),DiscussionsSideColumn.prototype.render=function render(){var e=this.props.className;return u.a.createElement("div",{className:a()("rc-DiscussionsSideColumn",e)},u.a.createElement(l.a,null),u.a.createElement(m.a,null),u.a.createElement(p.a,null))},DiscussionsSideColumn}(u.a.Component);d.propTypes={className:i.a.string},e.a=d},Edfi:function(module,exports,e){},EgRK:function(module,exports,e){var t=e("2l4Y"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},EyYU:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("sbe7"),a=t.n(s),o=t("GP6s"),i=t.n(o),c=t("STl7"),u=t("RzxB"),l=t.n(u),m=function(e){function DiscussionsLandingPageHeader(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsLandingPageHeader,e),DiscussionsLandingPageHeader.prototype.render=function render(){return a.a.createElement("div",{className:"rc-DiscussionsLandingPageHeader align-horizontal-center horizontal-box vertical-when-mobile"},a.a.createElement("div",{className:"flex-2 header-title"},a.a.createElement("h1",{className:"display-3-text"},i()("Discussion Forums")),a.a.createElement("p",{className:"body-1-text"},i()("Get help and discuss course material with the community."))),a.a.createElement("div",{className:"flex-1 align-self-end session-switcher-container"},a.a.createElement(c.a,null)))},DiscussionsLandingPageHeader}(a.a.Component);e.a=m},"Fe+3":function(module,exports,e){var t=e("uDV/"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},G92m:function(module,exports,e){},Hns6:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("sbe7"),i=t.n(o),c=t("Akn8"),u=t("wVBo"),l=t("ekry"),m=t("GP6s"),p=t.n(m),d=t("y1LI"),f=t("vpZN"),h=t("oQ9N"),b=t("3CU7"),y=t.n(b),v=function(e){function DiscussionsGroupForums(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsGroupForums,e),DiscussionsGroupForums.prototype.render=function render(){var e=this.props,t=e.groupForums,n=e.groupForumStatistics,r=e.courseSlug;if(!t||!t.length)return null;var s=t.map(function(e){var t=n.find(function(t){return t.id===e.id});return{id:e.id,discussionsLink:c.a.join(Object(f.c)(r),e.link),title:e.title,description:e.description,lastAnsweredAt:t&&t.lastAnsweredAt,forumQuestionCount:t&&t.forumQuestionCount}});return i.a.createElement(u.a,{className:"rc-DiscussionsGroupForums",title:i.a.createElement("span",null,i.a.createElement(d.a,{name:"lock"}),p()("Private Group Forums"))},i.a.createElement("ul",null,s.map(function(e){return i.a.createElement("li",{key:e.id},i.a.createElement(l.a,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))})))},DiscussionsGroupForums}(i.a.Component);e.a=a.a.compose(Object(h.a)({fields:["link","title","description","parentForumId"]}))(v)},IMZc:function(module,exports,e){var t=e("UxTF"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},N9MC:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("y1LI"),m=t("GP6s"),p=t.n(m),d=t("kvW3"),f=t("8cuT"),h=t.n(f),b=t("+LJP"),y=t("EgRK"),v=t.n(y),g=function(e){function LandingPageSearchResultsSummary(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).cancelSearchResults=function(){t.context.router.push({pathname:t.context.router.location.pathname,params:t.context.router.params,query:{}})},t}var t;return r()(LandingPageSearchResultsSummary,e),LandingPageSearchResultsSummary.prototype.render=function render(){if(!this.props.query)return null;return u.a.createElement("div",{className:"rc-LandingPageSearchResultsSummary horizontal-box align-items-spacebetween bgcolor-black-g1"},u.a.createElement("div",{className:"search-results"},u.a.createElement(d.a,{message:p()("{numResults} {numResults, plural,\n              =1 {search result} =0 {0 search results} other {search results}} for {query}"),numResults:this.props.numResults,query:u.a.createElement("strong",null,this.props.query)})),u.a.createElement("button",{onClick:this.cancelSearchResults,className:"nostyle cancel-button"},u.a.createElement(l.a,{name:"close",className:"color-secondary-text"})))},LandingPageSearchResultsSummary}(u.a.Component);g.propTypes={id:i.a.string,query:i.a.string,numResults:i.a.number},g.contextTypes={router:i.a.object.isRequired},e.a=Object(s.compose)(Object(b.a)(function(e){return{query:e.location.query.q&&decodeURIComponent(e.location.query.q)}}),h()(["DiscussionsSearchStore"],function(e,t){var n;return{numResults:e.DiscussionsSearchStore.getNumResults({forumId:t.id,query:t.query})}}))(g)},Qdyn:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("Akn8"),m=t("wVBo"),p=t("ekry"),d=t("BZ+2"),f=t("zFZo"),h=t("Fi0Z"),b=t("SaLm"),y=t("hN99"),v=t("oQ9N"),g=t("GP6s"),E=t.n(g),x=t("z+5C"),C=t.n(x),N=t("+9K8"),F=function(e){function DiscussionsCourseForums(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsCourseForums,e),DiscussionsCourseForums.prototype.render=function render(){var e=this.props,t=e.courseForums,n=e.mentorForums,r=e.courseForumStatistics,s=e.courseSlug,a=t.find(function(e){return e.forumType.typeName===b.j.rootForumType}),o=n.filter(function(e){return!e.parentForumId}),i=t.filter(function(e){return e.parentForumId===a.id&&e.forumType.typeName===b.j.customForumType}),c=o.concat(i).map(function(e){var t=r.find(function(t){return t.courseForumId===e.id});return Object.assign(e,{discussionsLink:l.a.join(Object(y.e)(s),e.link),lastAnsweredAt:t&&t.lastAnsweredAt,forumQuestionCount:t&&t.forumQuestionCount})});if(0===c.length)return null;return u.a.createElement(m.a,{title:E()("Discussion Forums"),className:"rc-DiscussionsCourseForums"},u.a.createElement("ul",null,c.map(function(e){return u.a.createElement("li",{key:e.id},u.a.createElement(p.a,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))})))},DiscussionsCourseForums}(u.a.Component);F.propTypes={courseForums:i.a.arrayOf(i.a.instanceOf(d.a)),mentorForums:i.a.arrayOf(i.a.instanceOf(f.a)),courseForumStatistics:i.a.arrayOf(i.a.instanceOf(h.a)),courseSlug:i.a.string},e.a=Object(s.compose)(Object(v.a)({fields:["link","title","description","lastAnsweredAt","forumQuestionCount","parentForumId"]}),N.a)(F)},Qnnn:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("c8Vh"),m=t("JUgE"),p=t("h4VP"),d=t("GP6s"),f=t.n(d),h=t("8cuT"),b=t.n(h),y=t("BZ+2"),v=t("oQ9N"),g=t("SaLm"),E=t("Fe+3"),x=t.n(E),C=function(e){function DiscussionsDescription(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsDescription,e),DiscussionsDescription.prototype.render=function render(){var e=u.a.createElement("p",null,f()("Welcome to the course discussion forums!\n            Ask questions, debate ideas, and find classmates who share your goals.\n            Browse popular threads below or other forums in the sidebar."));if(this.props.courseForums&&this.props.courseForums.length){var t=this.props.courseForums.find(function(e){return e.forumType.typeName===g.j.rootForumType});t&&!p.a.isEmpty(t.description)&&(e=u.a.createElement(m.a,{cml:t.description}))}return u.a.createElement(l.a,{className:"rc-DiscussionsDescription",showToggle:!1,isInitiallyCollapsed:!1},u.a.createElement("h2",{className:"label-text color-secondary-text"},f()("Description")),u.a.createElement("div",{className:"description"},e))},DiscussionsDescription}(u.a.Component);C.propTypes={courseForums:i.a.arrayOf(i.a.instanceOf(y.a))},e.a=Object(s.compose)(b()(["CourseStore"],function(e,t){var n=e.CourseStore;return{courseId:n.getCourseId(),courseSlug:n.getCourseSlug()}}),Object(v.a)({fields:["description","forumType"]}))(C)},Ras7:function(module,exports,e){},RzxB:function(module,exports,e){var t=e("COA5"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},To2G:function(module,exports,e){var t=e("eOCd"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},Up9C:function(module,exports,e){var t=e("ivlw"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},UxTF:function(module,exports,e){},"W/IG":function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("17x9"),a=t.n(s),o=t("sbe7"),i=t.n(o),c=t("BkbU"),u=t("GP6s"),l=t.n(u),m=t("3/q2"),p=t.n(m),d=function(e){function LandingPageSearchBox(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).state={query:t.props.query||""},t.onChange=function(e){t.setState({query:e.target.value})},t.onSubmit=function(){t.props.onSubmit(t.state.query)},t.checkForEnter=function(e){"Enter"===e.key&&t.onSubmit()},t}r()(LandingPageSearchBox,e);var t=LandingPageSearchBox.prototype;return t.componentWillReceiveProps=function componentWillReceiveProps(e){e.query!==this.props.query&&this.setState({query:e.query||""})},t.render=function render(){var e=this.state.query;return i.a.createElement("div",{className:"rc-LandingPageSearchBox"},i.a.createElement("div",{className:"search-bar"},i.a.createElement("div",{className:"input-area"},i.a.createElement(c.a,{trackingName:"search_box",placeholder:l()("Search"),className:"search-input",value:e,onChange:this.onChange,onKeyPress:this.checkForEnter,"aria-label":l()("Search Input")})),i.a.createElement("button",{type:"button",className:"search-button",onClick:this.onSubmit,"aria-label":l()("submit search")},i.a.createElement("i",{className:"cif-search","aria-hidden":"true"}))))},LandingPageSearchBox}(i.a.Component);d.propTypes={onSubmit:a.a.func.isRequired,query:a.a.string},e.a=d},"Xf+T":function(module,exports,e){},YmuN:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("TmOg"),a=t("VYij"),o=t.n(a),i=t("17x9"),c=t.n(i),u=t("sbe7"),l=t.n(u),m=t("w/1P"),p=t.n(m),d=t("eJMc"),f=t.n(d),h=t("GP6s"),b=t.n(h),y=t("+LJP"),v=t("hN99"),g=t("Up9C"),E=t.n(g),x=function(e){function TabList(){return e.apply(this,arguments)||this}var t;return r()(TabList,e),TabList.prototype.render=function render(){return l.a.createElement("ul",{className:"tabs tabs-divider horizontal-box",role:"tablist"},this.props.children,l.a.createElement("li",{className:"flex-1 align-right align-self-center rc-DiscussionsLandingPageSearch"},this.props.searchEl))},TabList}(l.a.Component);x.propTypes={searchEl:c.a.node,children:c.a.node};var C=Object(s.a)(x),N=function(e){function DiscussionsLandingPageTabs(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).onTabEnter=function(e){t.props.updateLocation(e["data-link"])},t.renderTabs=function(e){return e.map(function(e){return t.renderTab(e)})},t.renderTab=function(e){var t=e.title,n=e.pathname,r=e.query,s=e.isActive,a=e.key,o={query:r,pathname:n},i=a.toLowerCase().replace(/ /gi,"-");return l.a.createElement("li",{id:"tab-".concat(i),role:"tab","aria-controls":"panel-".concat(i),"aria-selected":s,className:p()("colored-tab",{selected:s}),"data-link":o,key:a},l.a.createElement(f.a,{to:o,tabIndex:-1},t))},t}var t;return r()(DiscussionsLandingPageTabs,e),DiscussionsLandingPageTabs.prototype.render=function render(){var e=[{title:b()("Forums"),pathname:Object(v.d)(),query:"",isActive:this.props.isBaseForumsActive,key:"Forums"},{title:b()("All Threads"),pathname:Object(v.c)(),query:this.props.query,isActive:!this.props.isBaseForumsActive,key:"All Threads"}];return l.a.createElement("div",{className:"rc-DiscussionsLandingPageTabs",tabIndex:-1},l.a.createElement(C,{onEnter:this.onTabEnter,searchEl:this.props.children},this.renderTabs(e)))},DiscussionsLandingPageTabs}(l.a.Component);N.propTypes={isBaseForumsActive:c.a.bool,query:c.a.object,children:c.a.node,updateLocation:c.a.func},e.a=Object(a.compose)(Object(y.a)(function(e){return{isBaseForumsActive:e.isActive({pathname:Object(v.d)()},!0),query:e.location.query,updateLocation:e.push}}))(N)},ZzvC:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("17x9"),a=t.n(s),o=t("sbe7"),i=t.n(o),c=t("i58+"),u=t("JUgE"),l=t("r3QC"),m=t("IDuc"),p=t("hN99"),d=t("8yDJ"),f=t("GP6s"),h=t.n(f),b=t("IMZc"),y=t.n(b),v=function(e){function DiscussionsWeekHeroUnit(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).goToForum=function(){var e;t.context.router.push(Object(p.j)(t.props.weekNumber))},t}var t;return r()(DiscussionsWeekHeroUnit,e),DiscussionsWeekHeroUnit.prototype.render=function render(){var e=this.props,t=e.weekNumber,n=e.isLoading,r=e.lastAnsweredAt,s=e.forumQuestionCount,a=e.title,o=e.description;if(!t)return null;return i.a.createElement(l.a,{className:"rc-DiscussionsWeekHeroUnit",label:h()("This Week's Forum")},n?i.a.createElement("div",{className:"message align-horizontal-center"},i.a.createElement("i",{className:"cif-spinner cif-spin cif-2x"})):i.a.createElement("div",{className:"horizontal-box vertical-when-mobile"},i.a.createElement("div",{className:"flex-1 left-column"},i.a.createElement("h2",{"aria-label":a,className:"headline-4-text"},a),i.a.createElement(u.a,{cml:o}),i.a.createElement(c.a,{lastAnsweredAt:r,forumQuestionCount:s})),i.a.createElement("div",{className:"align-self-center"},i.a.createElement(m.a,{className:"primary cozy",onClick:this.goToForum,trackingName:"discussions_hero_unit"},h()("Go to forum")))))},DiscussionsWeekHeroUnit}(i.a.Component);v.propTypes={title:a.a.string.isRequired,description:d.a.isRequired,weekNumber:a.a.number,isLoading:a.a.bool,lastAnsweredAt:a.a.number,forumQuestionCount:a.a.number},v.contextTypes={router:a.a.object.isRequired},e.a=v},dmYr:function(module,exports,e){var t=e("+o2K"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},eOCd:function(module,exports,e){},ekry:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("17x9"),a=t.n(s),o=t("sbe7"),i=t.n(o),c=t("eJMc"),u=t.n(c),l=t("bdFs"),m=t.n(l),p=t("y1LI"),d=t("kvW3"),f=t("8yDJ"),h=t("JUgE"),b=t("h4VP"),y=t("GP6s"),v=t.n(y),g=t("5fK1"),E=t.n(g),x=function(e){function ForumsLabel(){return e.apply(this,arguments)||this}var t;return r()(ForumsLabel,e),ForumsLabel.prototype.render=function render(){var e=this.props,t=e.discussionsLink,n=e.title,r=e.description,s=e.lastAnsweredAt,a=e.forumQuestionCount,o="number"==typeof a,c=!!s,l=1===a?v()("thread"):v()("threads"),f="Forum: ".concat(n,"\n                       ").concat(c?"Last Post: ".concat(m()(s).fromNow()):"","\n                       ").concat(o?"".concat(a," ").concat(l):"");return i.a.createElement(u.a,{className:"rc-ForumsLabel nostyle",to:t,"aria-label":f},i.a.createElement("div",{className:"forum-item horizontal-box align-items-vertical-center"},i.a.createElement("div",{className:"flex-1 forum-title-box"},i.a.createElement("p",{className:"headline-3-text"},n),!b.a.isEmpty(r)&&i.a.createElement(h.a,{cml:r}),c&&i.a.createElement("span",{className:"caption-text color-secondary-text"},i.a.createElement(d.b,{message:v()("Last post {timeAgo}"),timeAgo:m()(s).fromNow()}))),o&&i.a.createElement("div",{className:"threads-count vertical-box align-items-absolute-center"},i.a.createElement("span",{className:"headline-1-text"},a),i.a.createElement("span",{className:"caption-text color-secondary-text"},l)),i.a.createElement(p.a,{className:"chevron-icon",name:"chevron-right-thin"})))},ForumsLabel}(i.a.Component);x.propTypes={discussionsLink:a.a.string.isRequired,title:a.a.string.isRequired,description:f.a.isRequired,lastAnsweredAt:a.a.number,forumQuestionCount:a.a.number},e.a=x},"i58+":function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("bdFs"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("kvW3"),m=t("GP6s"),p=t.n(m),d=function(e){function WeekStats(){return e.apply(this,arguments)||this}var t;return r()(WeekStats,e),WeekStats.prototype.render=function render(){var e=this.props,t=e.lastAnsweredAt,n=e.forumQuestionCount;return u.a.createElement("div",{className:"rc-WeekStats caption-text color-secondary-text"},"number"==typeof n&&u.a.createElement(l.b,{message:p()("{forumQuestionCount} threads"),forumQuestionCount:n}),!!t&&u.a.createElement("span",null,"number"==typeof n&&u.a.createElement("span",null," · "),u.a.createElement(l.b,{message:p()("Last post {timeSinceLastPost}"),timeSinceLastPost:a()(t).fromNow()})))},WeekStats}(u.a.Component);d.propTypes={lastAnsweredAt:i.a.number,forumQuestionCount:i.a.number},e.a=d},ivlw:function(module,exports,e){},jplQ:function(module,e,t){"use strict";t.r(e);var n=t("VbXa"),r=t.n(n),s=t("sbe7"),a=t.n(s),o=t("VYij"),i=t.n(o),c=t("Qdyn"),u=t("7450"),l=t("1Ovw"),m=t("EyYU"),p=t("pFS+"),d=t("Dr+H"),f=t("lkFK"),h=t("oQ9N"),b=t("pKM5"),y=t.n(b),v=function(e){function DiscussionsLandingPage(){return e.apply(this,arguments)||this}var t;return r()(DiscussionsLandingPage,e),DiscussionsLandingPage.prototype.render=function render(){var e=this.props.courseId;return a.a.createElement("div",{className:"rc-DiscussionsLandingPage"},a.a.createElement(m.a,null),a.a.createElement(l.a,null),a.a.createElement("div",{className:"horizontal-box vertical-when-mobile"},a.a.createElement(p.a,{className:"flex-2"}),a.a.createElement(d.a,{className:"flex-1"})))},DiscussionsLandingPage}(a.a.Component);e.default=Object(o.compose)(Object(h.a)({fields:[],subcomponents:[c.a,u.a,f.a]}))(v)},lkFK:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("oVXz"),m=t("IAOZ"),p=t("Akn8"),d=t("oQ9N"),f=t("+LJP"),h=t("pqXt"),b=t("BZ+2"),y=t("N9MC"),v=t("dmYr"),g=t.n(v),E=function(e){function LandingPageThreadsViewWrapper(){return e.apply(this,arguments)||this}var t;return r()(LandingPageThreadsViewWrapper,e),LandingPageThreadsViewWrapper.prototype.render=function render(){if(!this.props.currentForum)return null;var e=p.a.join(m.d.learnRoot,this.props.courseSlug,this.props.currentForum.link);return u.a.createElement("div",{className:"rc-LandingPageThreadsViewWrapper"},u.a.createElement(y.a,{id:this.props.currentForum.forumId}),u.a.createElement(l.a,{backLink:e}))},LandingPageThreadsViewWrapper}(u.a.Component);E.propTypes={search:i.a.string,currentForum:i.a.instanceOf(b.a),courseSlug:i.a.string},e.a=Object(s.compose)(Object(d.a)({fields:["link"]}),Object(f.a)(h.a))(E)},lwDP:function(module,exports,e){},o3Oq:function(module,exports,e){},"pFS+":function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("w/1P"),i=t.n(o),c=t("17x9"),u=t.n(c),l=t("sbe7"),m=t.n(l),p=t("W/IG"),d=t("Qdyn"),f=t("7450"),h=t("YmuN"),b=t("lkFK"),y=t("Hns6"),v=t("hN99"),g=t("+LJP"),E=t("8wD4"),x=t.n(E),C=function(e){function DiscussionsMainColumn(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).onSubmitSearch=function(e){t.context.router.push({pathname:Object(v.c)(),query:{q:e}})},t}var t;return r()(DiscussionsMainColumn,e),DiscussionsMainColumn.prototype.render=function render(){var e=this.props,t=e.className,n=e.isForumsListActive,r=e.query;return m.a.createElement("div",{className:i()("rc-DiscussionsMainColumn",t)},n&&m.a.createElement(y.a,null),m.a.createElement(h.a,null,m.a.createElement(p.a,{query:r,onSubmit:this.onSubmitSearch})),n?m.a.createElement("div",null,m.a.createElement("div",{role:"tabpanel","aria-labelledby":"tab-forums",id:"panel-forums",tabIndex:0},m.a.createElement(d.a,null),m.a.createElement(f.a,null)),m.a.createElement("div",{id:"panel-all-threads"})):m.a.createElement("div",{role:"tabpanel","aria-labelledby":"tab-all-threads",id:"panel-all-threads",tabIndex:0},m.a.createElement(b.a,null)))},DiscussionsMainColumn}(m.a.Component);C.propTypes={className:u.a.string,isForumsListActive:u.a.bool,query:u.a.string},C.contextTypes={router:u.a.object.isRequired},e.a=Object(s.compose)(Object(g.a)(function(e){return{isForumsListActive:e.isActive({pathname:Object(v.d)()},!0),query:e.location.query.q&&decodeURIComponent(e.location.query.q)}}))(C)},pKM5:function(module,exports,e){var t=e("Ras7"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"uDV/":function(module,exports,e){},w1w6:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("VYij"),a=t.n(s),o=t("17x9"),i=t.n(o),c=t("sbe7"),u=t.n(c),l=t("Rto6"),m=t("GP6s"),p=t.n(m),d=t("8cuT"),f=t.n(d),h=t("wdve"),b=t("9A5E"),y=t("bN4z"),v=t("To2G"),g=t.n(v),E=function(e){function DiscussionsModerators(){for(var t,n=arguments.length,r=new Array(n),s=0;s<n;s++)r[s]=arguments[s];return(t=e.call.apply(e,[this].concat(r))||this).onDisplayMiniProfile=function(e){if(e===t.openMiniProfile)return;void 0!==t.openMiniProfile&&t.openMiniProfile.hide({}),t.openMiniProfile=e},t}r()(DiscussionsModerators,e);var t=DiscussionsModerators.prototype;return t.componentWillMount=function componentWillMount(){var e=this.props.courseId;this.props.staffSocialProfiles.length||this.context.executeAction(h.b,{courseId:e})},t.render=function render(){var e=this,t;if(!this.props.staffSocialProfiles)return null;if(!this.props.staffSocialProfiles.find(function(e){return e.courseRole===y.d.MENTOR||e.courseRole===y.d.TEACHING_STAFF||e.courseRole===y.d.COURSE_ASSISTANT}))return null;return u.a.createElement("div",{className:"rc-DiscussionsModerators card-no-action cozy"},u.a.createElement("div",{className:"label-text color-secondary-text"},p()("Moderators")),u.a.createElement("ul",{className:"moderator-list"},this.props.staffSocialProfiles.map(function(t){return u.a.createElement("li",{className:"staff-profile-container",key:t.id},u.a.createElement(l.a,{onDisplayMiniProfile:e.onDisplayMiniProfile,externalUserId:t.externalUserId,fullName:t.fullName,profileImageUrl:t.photoUrl,courseRole:t.courseRole}))})),u.a.createElement("div",{className:"caption-text horizontal-box align-items-absolute-center"},u.a.createElement(b.a,{href:"https://learner.coursera.help/hc/articles/208280006",trackingName:"mentor_learn_more",target:"_blank",className:"text-primary"},p()("Learn more about becoming a Mentor"))))},DiscussionsModerators}(u.a.Component);E.propTypes={staffSocialProfiles:i.a.arrayOf(i.a.object),courseId:i.a.string},E.contextTypes={executeAction:i.a.func.isRequired},e.a=Object(s.compose)(f()(["CourseStore","ClassmatesProfileStore"],function(e){var t=e.CourseStore,n=e.ClassmatesProfileStore;return{courseId:t.getCourseId(),staffSocialProfiles:n.getStaffProfiles()}}))(E)},wVBo:function(module,e,t){"use strict";var n=t("VbXa"),r=t.n(n),s=t("17x9"),a=t.n(s),o=t("sbe7"),i=t.n(o),c=t("w/1P"),u=t.n(c),l=t("yOG0"),m=t.n(l),p=function(e){function ForumsList(){return e.apply(this,arguments)||this}var t;return r()(ForumsList,e),ForumsList.prototype.render=function render(){return i.a.createElement("div",{className:u()(this.props.className,"rc-ForumsList","card-no-action")},i.a.createElement("h2",{className:"label-text color-secondary-text category-title"},this.props.title),i.a.createElement("div",null,this.props.children))},ForumsList}(i.a.Component);p.propTypes={title:a.a.node.isRequired,className:a.a.string,children:a.a.node},e.a=p},"x9/O":function(module,exports,e){},yOG0:function(module,exports,e){var t=e("lwDP"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)},"z+5C":function(module,exports,e){var t=e("Edfi"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var r={transform:void 0},s=e("aET+")(t,r);t.locals&&(module.exports=t.locals)}}]);
//# sourceMappingURL=71.ddc8eae39aaf5bbcc3c1.js.map