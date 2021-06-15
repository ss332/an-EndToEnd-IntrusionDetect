# unsw_nb15: 
> feature | 解释 |
>1. dur --记录的总持续时间
>2. proto --传输协议
>3. service --应用层协议 http ftp smtp ssh dns ftp-data -
>4. state --表明状态和使用的协议 ACC CLO CON ECO ECR FIN INT MAS PAR REQ RST TST TXD URH URN -
> 5. spkts -- 源地址到目标的包数目
> 6. dpkts -- 目的到源的包数目
> 7. sbytes --源到目标字节数
> 8. dbytes -- 目标到源字节数
> 9. rate -- 
> 10. sttl --源到目标的生存时间
> 11. dttl -- 目标到源生存时间
> 12. sload --源发送字节/s
> 13.dload --目标字节/s
> 14. sloss --重新传输和丢弃的源数据包
> 15. dloss --重新传输和丢弃的目标数据包
> 16. sinpkt -- 源中间包到达时间
> 17. dinpkt -- 目标中间包到达时间
> 18. sjit -- 源偏差 ms
> 19. djit -- 目标偏差 ms
> 20. swin --源tcp窗口通知大小
> 21. dwin --目标端口tcp窗口通知大小
> 22. stcpb --源tcp基础序列号
> 23. dtcpb -- 目标tcp基础序列号
> 24. tcprtt -- tcp建立往返使用时间，tcp的synack和ackdat之和
> 25. synack -- tcp连接建立时间,tcpSYN到tcpSYN_ACK数据包之间的时间
> 26. ackdat --  tcp连接建立时间,tcpSYN_ACK到ACK数据包之剑时间
> 27. smean -- 源发送的数据流包的平均大小
> 28. dmean -- 目标发送的数据包流平均大小
> 29. trans_depth -- 表示http请求/响应事务连接的管道化深度
> 30. response_body_len -- 从服务器的http服务传输的数据的实际未压缩内容大小。
> 31. ct_srv_src -- 在最近时间内100个连接中具有相同服务类型和源地址的个数
> 32. ct_state_ttl -- 在连接生存特定时间内state的数目
> 33. ct_dst_itm -- 在最近100个连接中具有相同目的地址的连接个数
> 34. ct_src_dport_item -- 在最近100个连接中具有相同源地址和目标端口号的连接个数
> 35. ct_dst_sport_itm -- 在最近100个连接中具有相同目标地址和源端口号的连接个数
> 36. ct_dst_src_itm -- 在最近100个连接中具有相同目标地址和源地址的连接个数
> 37. is_ftp_login -- ftp会话是否有用户输入用户名和密码登录发起的，是为1否则0
> 38. ct_ftp_cmd -- ftp会话中有命令的流的数目
> 39. ct_flw_http_mthd -- http服务中具有Get和Post等方法的流数
> 40. ct_src_itm -- 在最近100个连接中具有相同源地址的连接个数
> 41. ct_srv_dst -- 在最近100个连接中具有相同服务和目标地址的连接个数
> 42. is_sm_ips_ports -- 是否源地址和目标地址相同，源端口和目标端口号相同为1，否为0
> 43. attack_cat -- 攻击种类的名字，该数据集共有9种攻击类型，Fuzzers, Analysis, Backdoors, DoS Exploits, Generic, Reconnaissance, Shellcode and Worms
> 44. Label -- 0是正常流量，1是攻击记录流
