---
layout: post
title:  "【技能|SQL】Matrix Business Analysis"
categories: jekyll update
---
## 1.写在前面
终于把codecademy上面的SQL课程全部学完啦～撒花～

![](/img/post0601.jpg)

中途还报了几个bug，不知道会不会收到回复。
下面放上学习过程中记录的笔记。

## 2.附上一些SQL相关参考文档
[w3school](http://www.w3school.com.cn/sql/sql_join_inner.asp)

## 3.学习笔记
1.销量按日期统计并排序
```
SELECT date(ordered_at),count(1)
from orders
group by 1
order by 1;
# 1 refers the first coloum
```
2.链接两个表进行统计排序
```
SELECT DATE(ordered_at), round(sum(amount_paid),2)
FROM orders
join order_items on orders.id=order_items.order_id
# 如果要查找某个具体东西按天的销售额，则可以使用where item=''这样的写法
group by 1
order by 1;
#注意两表链接时需要找准相同的column，有时它们会有不同的命名方式
```
3.计算每种物品的销量，按照销量排序(降序排列)
```
SELECT name, round(sum(amount_paid),2)
from order_items
group by name
order by 2 desc;
```
4.计算每种物品销量的总占比，注意select的写法
```
SELECT name, round(sum(amount_paid)/(select sum(amount_paid) from order_items)*100.0,2) as asp
from order_items
group by name
order by 2 desc;
```
5.case 在使用中如果省略每次写列名的话，就要在前面统一写上，比如 case name
```
select
  case name
  #注意case的写法
    when 'kale-smoothie'    then 'smoothie'
    when 'banana-smoothie'  then 'smoothie'
    when 'orange-juice'     then 'drink'
    when 'soda'             then 'drink'
    when 'blt'              then 'sandwich'
    when 'grilled-cheese'   then 'sandwich'
    when 'tikka-masala'     then 'dinner'
    when 'chicken-parm'     then 'dinner'
    else 'other'
  end as category, round(1.0 * sum(amount_paid) /
    (select sum(amount_paid) from order_items) * 100, 2) as pct
from order_items
group by 1
order by 2 desc;
```
6. reorder rate
```
select name, round(1.0 * count(distinct order_id) /
  count(distinct orders.delivered_to), 2) as reorder_rate
from order_items
  join orders on
    orders.id = order_items.order_id
group by 1
order by 2 desc;
```
7.
### KPI
### daily revenue,DAU, Daily ARPU, Daily ARPPU,1 Day Retention

```
select
  date(created_at),
 count(distinct(user_id)) as dau
 from gameplays
 group by 1
 order by 1;
 计算DAU
 ```
 8.使用with语句来存储temporary 值
 ```
 with  daily_revenue as (
select
date(created_at) as dt, 
round(sum(price),2) as rev
  from purchases
  where refunded_at is null
  group by 1
)
select *
from daily_revenue order by dt;
```
一个复杂的链接两表的例子
```
with  daily_revenue as (
select
date(created_at) as dt, 
round(sum(price),2) as rev
  from purchases
  where refunded_at is null
  group by 1
),
daily_players as (
 select
  date(created_at) as dt,
  count(distinct(user_id)) as players
  from gameplays
  group by 1
)
select 
daily_revenue.dt,
daily_revenue.rev/daily_players.players
from daily_revenue
join daily_players on daily_revenue.dt=daily_players.dt;
```
==不用with的办法（正确性未知）==
```
select date(gameplays.created_at) as dt,
round(sum(select purchases.price from purchases where refunded_at is null)/count(distinct(gameplays.user_id)),2)  as apru
from purchases
join gameplays on gameplays.date(created_at)=purchases.date(created_at)
group by 1
order by 1;
```
9.using
```
from daily_revenue
join daily_players using (dt)
# 等价于
from daily_revenue
join daily_players on
daily_revenue.dt=daily_players.dt;
```
留存率需要使用到self-join
```
select
date(g1.created_at) as dt,
g1.user_id
from gameplays as g1
join gameplays as g2 on
g1.user_id=g2.user_id
order by 1
limit 100;
```
使用left join来得到更有意义的数据
```
select
  date(g1.created_at) as dt,
  round(100 * count(distinct g2.user_id) /
    count(distinct g1.user_id)) as retention
from gameplays as g1
  left join gameplays as g2 on
    g1.user_id = g2.user_id
    and date(g1.created_at) = date(datetime(g2.created_at, '-1 day'))
group by 1
order by 1
limit 100;
```
